import numpy as np
import math
from numba import njit, prange
import numba as nb
import os
import matplotlib.pyplot as plt

# Stałe fizyczne
MSUN = 1.98847e30  # kg, masa Słońca
RSUN = 6.957e8  # m, promień Słońca
AU = 1.495978707e11  # m
YEAR_S = 365.25 * 24.0 * 3600.0  # s
TUNIT = YEAR_S / (2.0 * math.pi)


# -----------------------
# Masa -> promień / temperatura
# -----------------------

def mass_to_radius(m_rel):
    """Przekształć masę (M_sun) na promień (AU).

    Przyjmuje skalar lub tablicę mas (M_sun). Zwraca skalar dla skalaru
    lub `np.ndarray` dla tablicy.
    """
    m_arr = np.atleast_1d(np.asarray(m_rel, dtype=float))
    R_au = mass_to_radius_jit(m_arr)
    if np.isscalar(m_rel):
        return float(R_au[0])
    return R_au


@njit(fastmath=True)
def mass_to_radius_jit(m_arr):
    n = m_arr.size
    out = np.empty(n, dtype=np.float64)
    e_low = 0.8
    e_high = 0.57
    center = 0.5
    width = 0.25
    for idx in range(n):
        m = m_arr[idx]
        x = math.log10(m)
        s = 0.5 * (1.0 + math.tanh((x - center) / width))
        exponent = e_low * (1.0 - s) + e_high * s
        correction = 1.0
        if x > 1.0:
            correction = 1.0 + 0.1 * math.log(m / 10.0)
        R_m = RSUN * (m ** exponent) * correction
        out[idx] = R_m / AU
    return out


def mass_to_temperature(m):
    """Przybliżenie temperatury efektywnej (K) z masy w M_sun.

    Działa dla skalaru i tablicy.
    """
    if np.ndim(m) == 0:
        return float(5778.0 * (float(m) ** 0.5))
    m_arr = np.asarray(m, dtype=float)
    return mass_to_temperature_jit(m_arr)


@njit(fastmath=True)
def mass_to_temperature_jit(m_arr):
    n = m_arr.size
    out = np.empty(n, dtype=np.float64)
    T_sun = 5778.0
    exponent = 0.5
    for i in range(n):
        out[i] = T_sun * (m_arr[i] ** exponent)
    return out


# -----------------------
# Siły i integrator
# -----------------------

@njit(parallel=True, fastmath=True)
def compute_accelerations(positions, masses, radii, G, out, eps=0.0, block=2, softening=0.0):
    N, d = positions.shape
    T = nb.get_num_threads()
    local = np.zeros((T, N, d))

    nblocks = max(1, (N + block - 1) // block)

    for bbi in prange(0, nblocks):
        tid = nb.get_thread_id()
        bi = bbi * block
        bi_end = bi + block
        if bi_end > N:
            bi_end = N

        bj = bi
        while bj < N:
            bj_end = bj + block
            if bj_end > N:
                bj_end = N

            if bj == bi:
                for i in range(bi, bi_end):
                    for j in range(i+1, bj_end):
                        r2 = 0.0
                        for k in range(d):
                            diff = positions[j, k] - positions[i, k]
                            r2 += diff * diff
                        soft2 = 0.0
                        if softening > 0.0:
                            s = softening * (radii[i] + radii[j])
                            soft2 = s * s
                        r2_eps = r2 + eps + soft2
                        pref = G / (r2_eps * math.sqrt(r2_eps))
                        for k in range(d):
                            diff = positions[j, k] - positions[i, k]
                            local[tid, i, k] += pref * masses[j] * diff
                            local[tid, j, k] -= pref * masses[i] * diff
            else:
                for i in range(bi, bi_end):
                    for j in range(bj, bj_end):
                        r2 = 0.0
                        for k in range(d):
                            diff = positions[j, k] - positions[i, k]
                            r2 += diff * diff
                        soft2 = 0.0
                        if softening > 0.0:
                            s = softening * (radii[i] + radii[j])
                            soft2 = s * s
                        r2_eps = r2 + eps + soft2
                        pref = G / (r2_eps * math.sqrt(r2_eps))
                        for k in range(d):
                            diff = positions[j, k] - positions[i, k]
                            local[tid, i, k] += pref * masses[j] * diff
                            local[tid, j, k] -= pref * masses[i] * diff

            bj += block

    for i in range(N):
        for k in range(d):
            out[i, k] = 0.0

    for t in range(T):
        for i in range(N):
            for k in range(d):
                out[i, k] += local[t, i, k]


@njit(fastmath=True)
def verlet_step(pos, vel, acc, dt, masses, radii, G, tmp_pos, tmp_acc, softening=0.0):
    N, d = pos.shape
    for i in range(N):
        for k in range(d):
            tmp_pos[i, k] = pos[i, k] + vel[i, k] * dt + 0.5 * acc[i, k] * (dt * dt)

    compute_accelerations(tmp_pos, masses, radii, G, tmp_acc, 1e-30, 64, softening)

    for i in range(N):
        for k in range(d):
            vel[i, k] += 0.5 * (acc[i, k] + tmp_acc[i, k]) * dt
            pos[i, k] = tmp_pos[i, k]
            acc[i, k] = tmp_acc[i, k]


@njit(fastmath=True)
def verlet_step_block(pos, vel, acc, dt, masses, radii, G, tmp_pos, tmp_acc, softening=0.0, block=64):
    """Verlet integrator variant that accepts a `block` parameter forwarded to compute_accelerations.

    This mirrors `verlet_step` but allows benchmarking different `block` sizes.
    """
    N, d = pos.shape
    for i in range(N):
        for k in range(d):
            tmp_pos[i, k] = pos[i, k] + vel[i, k] * dt + 0.5 * acc[i, k] * (dt * dt)

    compute_accelerations(tmp_pos, masses, radii, G, tmp_acc, 1e-30, block, softening)

    for i in range(N):
        for k in range(d):
            vel[i, k] += 0.5 * (acc[i, k] + tmp_acc[i, k]) * dt
            pos[i, k] = tmp_pos[i, k]
            acc[i, k] = tmp_acc[i, k]


@njit(fastmath=True)
def yoshida_step(pos, vel, acc, dt, masses, radii, tmp_pos, tmp_acc, softening=0.0, G=1):
    w1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    w0 = -2.0**(1.0/3.0) * w1

    verlet_step(pos, vel, acc, w1 * dt, masses, radii, G, tmp_pos, tmp_acc, softening)
    verlet_step(pos, vel, acc, w0 * dt, masses, radii, G, tmp_pos, tmp_acc, softening)
    verlet_step(pos, vel, acc, w1 * dt, masses, radii, G, tmp_pos, tmp_acc, softening)


# -----------------------
# Funkcje pomocnicze do inicjalizacji
# -----------------------

def random_masses(N, min_mass=0.08, max_mass=100):
    def xi(M):
        m = np.atleast_1d(np.asarray(M, dtype=float))
        val = np.empty_like(m)
        mask = m < 1.0
        if mask.any():
            val[mask] = 0.158/(m[mask]*np.log(10.0)) * np.exp(-((np.log10(m[mask]) - np.log10(0.08))**2) / (2.0 * 0.69**2))
        if (~mask).any():
            val[~mask] = 0.0195 * (m[~mask] ** -2.3)
        return val

    masses = []
    xs_grid = np.logspace(np.log10(min_mass), np.log10(max_mass), 1000)
    ymax = float(np.max(xi(xs_grid)))
    chunk = 4096
    while len(masses) < N:
        size = min(chunk, N - len(masses))
        Ms = np.random.uniform(min_mass, max_mass, size=size)
        Ys = np.random.uniform(0.0, ymax, size=size)
        mask = Ys < xi(Ms)
        if np.any(mask):
            masses.extend(Ms[mask].tolist())

    masses = np.array(masses[:N])

    save_dir = os.path.join(os.getcwd(), 'plots')
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    nbins = 50
    bin_edges = np.logspace(np.log10(min_mass), np.log10(max_mass), nbins + 1)
    plt.hist(masses, bins=bin_edges, density=True, alpha=0.6, color='g', label='Generated masses')

    xs = np.logspace(np.log10(min_mass), np.log10(max_mass), 2000)
    pdf = xi(xs)
    if xs.size >= 2:
        integral = np.sum(0.5 * (pdf[:-1] + pdf[1:]) * (xs[1:] - xs[:-1]))
    else:
        integral = float(np.sum(pdf))
    if integral > 0.0:
        pdf = pdf / integral
    plt.plot(xs, pdf, color='r', label='Chabrier IMF (normalized)')
    plt.xscale('log')
    plt.xlabel('Mass [M_sun]')
    plt.ylabel('Probability Density [1 / M_sun]')
    plt.title('Initial Mass Function (Chabrier IMF)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(save_dir, 'mass_distribution.png'), dpi=150)
    plt.close()
    return masses


def create_rotating_disk(N, masses, r_min=1e10/AU, r_max=5e11/AU, G=1, v_scale=0.95, inward_fraction=0.02, clockwise=True, seed=None):
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    r = np.sqrt(u * (r_max**2 - r_min**2) + r_min**2)
    theta = rng.random(N) * 2.0 * np.pi

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    positions = np.column_stack((x, y)).astype(float)

    order = np.argsort(r)
    m_sorted = masses[order]
    cumsum = np.cumsum(m_sorted)
    M_enc = np.empty(N, dtype=float)
    M_enc[order] = cumsum

    safe_r = np.maximum(r, 1e-12)
    v_circ = np.sqrt(np.maximum(G * M_enc, 0.0) / safe_r)

    if clockwise:
        t_x = np.sin(theta)
        t_y = -np.cos(theta)
    else:
        t_x = -np.sin(theta)
        t_y = np.cos(theta)

    r_x = np.cos(theta)
    r_y = np.sin(theta)

    vt = (v_scale * v_circ).reshape(-1, 1) * np.column_stack((t_x, t_y))
    vr_in = -(inward_fraction * v_circ).reshape(-1, 1) * np.column_stack((r_x, r_y))

    velocities = (vt + vr_in).astype(float)
    return positions, velocities
