import time
import math
import argparse
import numpy as np
import time
import argparse
import numpy as np
import numba as nb
import sys

from physics import compute_accelerations, verlet_step_block


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark simulation timings with parameter sweeps")
    p.add_argument("--threads", "-t", type=int, default=18, help="Number of numba threads to use")
    p.add_argument("--block", "-b", type=int, default=0, help="If >0 use this block for all runs; otherwise default block = max(1, N//threads)")
    p.add_argument("--param", "-p", action="append", help="Parameter sweep definition: name=vals. vals can be comma list or start:stop:count. Use up to two \"-p\" entries for a 2D grid. Example: -p N=10,50,100 -p dt=0.001:0.01:5")
    p.add_argument("--list-params", action="store_true", help="List available parameter names for -p and exit")
    p.add_argument("--list-args", action="store_true", help="List available CLI arguments (names only) and exit")
    p.add_argument("--base-N", type=int, default=50, help="Base N to use when N is not being swept")
    p.add_argument("--steps", "-s", type=int, default=20000, help="Total integrator steps per run (stops early if --time-limit is reached)")
    p.add_argument("--time-limit", type=float, default=0.0, help="Optional per-run time limit in seconds (0=no limit)")
    p.add_argument("--dt", type=float, default=0.01, help="Base timestep to use when dt is not being swept")
    p.add_argument("--dim", type=int, default=2, help="Dimension of the simulation (2 or 3)")
    p.add_argument("--seed", type=int, default=12345, help="RNG seed base for deterministic initial conditions")
    p.add_argument("--save", type=str, default="", help="Optional .npz path to save results (for 1D or 2D sweeps)")
    return p.parse_args()


def parse_values(s):
    s = s.strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 3:
            a, b, cnt = parts
            a = float(a); b = float(b); cnt = int(cnt)
            return np.linspace(a, b, cnt)
        elif len(parts) == 2:
            a, b = parts
            return np.linspace(float(a), float(b), 10)
        else:
            raise ValueError("Invalid range format, use start:stop:count or start:stop")
    else:
        parts = [x for x in s.split(",") if x.strip()]
        return np.array([float(x) for x in parts])


def cast_values_by_name(name, arr):
    name = name.lower()
    if name in ("n", "N", "nbodies", "n_bodies"):
        return arr.astype(int)
    if name in ("steps", "iterations", "steps_per_frame"):
        return arr.astype(int)
    if name in ("block", "threads"):
        return arr.astype(int)
    return arr.astype(float)


def generate_inputs(N, dim=2, seed=12345):
    rng = np.random.default_rng(seed + int(N))
    positions = rng.random((N, dim)).astype(np.float64)
    masses = rng.uniform(0.1, 10.0, size=N).astype(np.float64)
    radii = np.zeros(N, dtype=np.float64)
    return positions, masses, radii


def simulate_and_time(N, dt, steps, block, threads, time_limit, dim=2, G=1.0, softening=0.0, seed=12345):
    nb.set_num_threads(threads)
    pos0, masses, radii = generate_inputs(N, dim, seed)
    vel0 = np.zeros_like(pos0)
    acc0 = np.zeros_like(pos0)
    tmp_pos = np.zeros_like(pos0)
    tmp_acc = np.zeros_like(acc0)

    # initial acceleration
    compute_accelerations(pos0, masses, radii, G, acc0, 1e-30, block, softening)

    # warmup compile (use copies so originals remain pristine)
    try:
        verlet_step_block(pos0.copy(), vel0.copy(), acc0.copy(), dt, masses, radii, G, tmp_pos, tmp_acc, softening, block)
    except Exception:
        pass

    # reset to initial
    pos = pos0.copy()
    vel = vel0.copy()
    acc = acc0.copy()

    t0 = time.perf_counter()
    steps_done = 0
    for i in range(steps):
        verlet_step_block(pos, vel, acc, dt, masses, radii, G, tmp_pos, tmp_acc, softening, block)
        steps_done += 1
        if time_limit > 0.0 and (time.perf_counter() - t0) >= time_limit:
            break
    elapsed = time.perf_counter() - t0
    avg = elapsed / steps_done if steps_done > 0 else float("inf")
    return {"N": N, "dt": dt, "steps_requested": steps, "steps_done": steps_done, "block": block, "threads": threads, "elapsed": elapsed, "avg_per_step": avg}


def main():
    args = parse_args()

    requested = args.threads
    try:
        nb.set_num_threads(requested)
        threads = nb.get_num_threads()
        print("Numba threads set to:", threads)
    except Exception as e:
        threads = nb.get_num_threads()
        if threads < 1:
            threads = 1
        print(f"requested {requested}, using {threads} (reason: {e})")

    if args.list_params:
        print("Available -p parameter names and behavior:")
        print(" - N, n, nbodies, n_bodies : integer - number of bodies")
        print(" - dt                    : float   - timestep per integrator step")
        print(" - steps, iterations, steps_per_frame : integer - steps per run (use --steps for global default; per-param steps not applied automatically)")
        print(" - block                 : integer - block size for compute_accelerations (use --block to set globally; per-param block not applied automatically)")
        print(" - threads               : integer - numba threads (use --threads to set; per-param threads not applied automatically)")
        print("Notes: The script currently treats 'N' and 'dt' specially when passed via -p. Unknown numeric names are interpreted as integer->steps or float->dt. Up to two -p entries allowed for a 2D sweep.")
        sys.exit(0)

    if args.list_args:
        print("Available CLI arguments (names only):")
        print(" - --threads, -t")
        print(" - --block, -b")
        print(" - --param, -p")
        print(" - --list-params")
        print(" - --list-args")
        print(" - --base-N")
        print(" - --steps, -s")
        print(" - --time-limit")
        print(" - --dt")
        print(" - --dim")
        print(" - --seed")
        print(" - --save")
        print("Use -h/--help for full descriptions of each option.")
        sys.exit(0)

    # parse params
    param_defs = args.param or []
    params = []
    for pstr in param_defs:
        if "=" not in pstr:
            print(f"Invalid --param '{pstr}', expected name=vals")
            sys.exit(1)
        name, valstr = pstr.split("=", 1)
        vals = parse_values(valstr)
        vals = cast_values_by_name(name, vals)
        params.append((name, vals))

    # default sweep if none provided: N sweep
    if len(params) == 0:
        params = [("N", np.array([3, 10, 50, 120], dtype=int))]

    if len(params) > 2:
        print("This script supports sweeping up to two parameters at once.")
        sys.exit(1)

    save_path = args.save if args.save else None
    results = None

    # single-parameter sweep
    if len(params) == 1:
        name, vals = params[0]
        out = []
        for v in vals:
            if name.lower() in ("n", "nbodies", "n_bodies", "N"):
                N = int(v)
                dt = args.dt
            elif name.lower() in ("dt",):
                N = args.base_N
                dt = float(v)
            else:
                # unknown param controls dt if float, or steps if integer
                if float(v).is_integer():
                    N = args.base_N
                    dt = args.dt
                else:
                    N = args.base_N
                    dt = float(v)

            block = args.block if args.block > 0 else max(1, N // threads)
            r = simulate_and_time(N, dt, args.steps, block, threads, args.time_limit, dim=args.dim, seed=args.seed)
            out.append(r)

        results = out
        # print summary
        print("\nSummary:")
        for r in results:
            print(f"{r['N']:4d} bodies, dt={r['dt']:.6g}, block={r['block']:3d}, steps={r['steps_done']:5d}, avg={r['avg_per_step']*1e3:8.6f} ms")

        if save_path:
            # convert to arrays
            Ns = np.array([r['N'] for r in results])
            avgs = np.array([r['avg_per_step'] for r in results])
            np.savez(save_path, names=Ns, avgs=avgs)
            print('Saved results to', save_path)

    else:
        # two-parameter sweep -> build 2D grid
        (name1, vals1), (name2, vals2) = params
        m = len(vals1)
        n = len(vals2)
        grid = np.zeros((m, n), dtype=np.float64)
        for i, v1 in enumerate(vals1):
            for j, v2 in enumerate(vals2):
                # determine N and dt depending on names
                if name1.lower() in ("n", "nbodies", "n_bodies", "N"):
                    N = int(v1)
                elif name2.lower() in ("n", "nbodies", "n_bodies", "N"):
                    N = int(v2)
                else:
                    N = args.base_N

                if name1.lower() == "dt":
                    dt = float(v1)
                elif name2.lower() == "dt":
                    dt = float(v2)
                else:
                    dt = args.dt

                block = args.block if args.block > 0 else max(1, N // threads)
                r = simulate_and_time(N, dt, args.steps, block, threads, args.time_limit, dim=args.dim, seed=args.seed)
                grid[i, j] = r['avg_per_step']
                print(f"[{i},{j}] N={N}, dt={dt:.6g} -> avg {r['avg_per_step']*1e3:.6f} ms")

        results = {"vals1_name": name1, "vals2_name": name2, "vals1": vals1, "vals2": vals2, "grid": grid}
        print("\n2D grid avg per-step (ms):")
        print((grid * 1e3))
        if save_path:
            np.savez(save_path, vals1=vals1, vals2=vals2, grid=grid)
            print('Saved 2D results to', save_path)


if __name__ == "__main__":
    main()
