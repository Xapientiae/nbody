import numpy as np
import math
import time
import threading
import queue
import os
from vispy import app, scene

from physics import MSUN, AU, TUNIT, mass_to_radius, mass_to_temperature, compute_accelerations, yoshida_step
from utils import masses_to_rgba, temperature_to_rgba


_HEADLESS_VISPY_BACKENDS = {'egl', 'osmesa', '_test', 'jupyter_rfb'}
_WINDOW_BACKEND_CANDIDATES = ('pyside6', 'pyqt6', 'pyqt5', 'pyside2', 'glfw', 'pyglet', 'sdl2', 'wx', 'tkinter')


def _ensure_window_backend():
    """Select a GUI backend explicitly.

    Some Linux setups auto-select EGL (headless), which runs timers but shows no window.
    """
    active = app.use_app()
    active_name = str(getattr(active, 'backend_name', '')).lower()
    if active_name and active_name not in _HEADLESS_VISPY_BACKENDS:
        return active_name

    errors = []
    for candidate in _WINDOW_BACKEND_CANDIDATES:
        try:
            selected = app.use_app(candidate)
            selected_name = str(getattr(selected, 'backend_name', candidate)).lower()
            if selected_name not in _HEADLESS_VISPY_BACKENDS:
                return selected_name
        except Exception as exc:
            errors.append(f'{candidate}: {exc}')

    details = '; '.join(errors) if errors else 'no backend candidates were tried'
    raise RuntimeError(
        'VisPy could not select a window-capable backend. '
        'Install one (recommended: PySide6) and retry. '
        f'Backend probe details: {details}'
    )


class NBodySimulation:
    """N-body visualizer with background simulation thread.

    The simulator publishes the newest full state to an internal `queue.Queue(maxsize=1)`
    and the Vispy main thread will render whatever is available when it draws.
    """

    def __init__(self, positions, velocities, masses, radii, G, dt, physics_steps_per_frame=1, marker_base=8.0, marker_scale_exponent=1.0, marker_max_size=80.0, marker_world_scaling=False, marker_world_scale=1.0, enable_collisions=True, density=5500.0, softening=0.1, collision_merge_margin=1.0, use_mass_radius=False, max_sim_steps_per_second=None, render_fps=60, leave_one_core=True, numba_threads=None, stats_log_interval=None, show_canvas=True, adaptive_steps_per_frame=False, min_physics_steps_per_frame=1, max_physics_steps_per_frame=64):
        self.window_backend = _ensure_window_backend()

        # core state (owned and mutated by simulation thread)
        self.pos = positions.copy()
        self.vel = velocities.copy()
        self.mass = masses
        self.radius = radii

        # shift to center-of-mass frame
        total_mass = np.sum(self.mass)
        com_pos = np.sum(self.mass[:, np.newaxis] * self.pos, axis=0) / total_mass
        self.pos -= com_pos
        com_velocity = np.sum(self.mass[:, np.newaxis] * self.vel, axis=0) / total_mass
        self.vel -= com_velocity

        # physics parameters
        self.G = G
        self.dt = dt
        self.enable_collisions = bool(enable_collisions)
        self.density = float(density)
        self.use_mass_radius = bool(use_mass_radius)
        self.softening = float(softening)
        self.collision_merge_margin = float(collision_merge_margin)

        # rendering / marker params
        self.marker_base = float(marker_base)
        self.marker_scale_exponent = float(marker_scale_exponent)
        self.marker_max_size = float(marker_max_size)
        self.marker_world_scaling = bool(marker_world_scaling)
        self.marker_world_scale = float(marker_world_scale)

        # simulation control
        self.physics_steps_per_frame = int(max(1, physics_steps_per_frame))
        self.max_sim_steps_per_second = float(max_sim_steps_per_second) if (max_sim_steps_per_second is not None and max_sim_steps_per_second > 0) else None
        self.adaptive_steps_per_frame = bool(adaptive_steps_per_frame)
        self.min_physics_steps_per_frame = int(max(1, min_physics_steps_per_frame))
        self.max_physics_steps_per_frame = int(max(self.min_physics_steps_per_frame, max_physics_steps_per_frame))
        self.physics_steps_per_frame = int(min(max(self.physics_steps_per_frame, self.min_physics_steps_per_frame), self.max_physics_steps_per_frame))
        self._adapt_every_batches = 20
        self._next_adapt_batch = self._adapt_every_batches

        # optionally restrict numba threads to leave one core free (helps UI responsiveness)
        self.leave_one_core = bool(leave_one_core)
        self.numba_threads = None
        if numba_threads is not None:
            try:
                import numba as _nb
                nt = int(numba_threads)
                if nt > 0:
                    _nb.set_num_threads(nt)
                    self.numba_threads = nt
            except Exception:
                pass
        elif self.leave_one_core:
            try:
                import numba as _nb
                import os as _os
                ncpu = _os.cpu_count() or 1
                nt = max(1, int(ncpu) - 1)
                _nb.set_num_threads(nt)
                self.numba_threads = nt
            except Exception:
                self.numba_threads = None

        # working arrays for integrator (owned by sim thread)
        self.acc = np.zeros_like(self.pos)
        self._tmp_pos = np.empty_like(self.pos)
        self._tmp_acc = np.empty_like(self.pos)

        # initialize accelerations
        compute_accelerations(self.pos, self.mass, self.radius, self.G, self.acc, 1e-30, 64, self.softening)

        # history / time
        self.energy_history = []
        self.momentum_history = []
        self.time_history = []
        self._time = 0.0

        if self.enable_collisions:
            try:
                self.resolve_collisions()
            except Exception:
                pass
        self.measure()

        # prepare marker sizes and colors (initial snapshot)
        self._radius_ref = self._get_radius_reference()
        self.compute_sizes()
        try:
            self.colors = masses_to_rgba(self.mass)
        except Exception:
            self.colors = np.ones((len(self.mass), 4), dtype=np.float32)

        # vispy canvas and scatter (render must run on main thread)
        self.canvas = scene.SceneCanvas(keys='interactive', show=show_canvas)
        if show_canvas:
            try:
                self.canvas.show()
            except Exception:
                pass
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'panzoom'

        pos_min = self.pos.min(axis=0)
        pos_max = self.pos.max(axis=0)
        margin = 0.2 * (pos_max - pos_min + 1e-6)
        self.view.camera.rect = (
            pos_min[0] - margin[0],
            pos_min[1] - margin[1],
            (pos_max - pos_min + 2*margin)[0],
            (pos_max - pos_min + 2*margin)[1]
        )

        self.compute_sizes()

        self.scatter = scene.visuals.Markers()
        self.scatter.set_data(self.pos, size=self.sizes, face_color=self.colors)
        self.view.add(self.scatter)

        # buffer pool: free buffers and a single-slot ready queue
        # allocate a small pool of reusable numpy buffers to avoid per-frame allocations
        self._buffer_pool = []
        pool_size = 2
        for i in range(pool_size):
            bp = {
                'id': i,
                'pos': np.empty_like(self.pos),
                'sizes': np.empty_like(self.sizes),
                'colors': np.empty_like(self.colors),
                'radii': np.empty_like(self.radius),
            }
            self._buffer_pool.append(bp)

        self._free_buffers = queue.Queue()
        for i in range(pool_size):
            self._free_buffers.put(i)

        # ready queue holds either (buf_idx, time) or
        # (pos_array, sizes_array, colors_array, radii_array, time)
        self._ready_queue = queue.Queue(maxsize=1)

        # publish an initial snapshot using a pooled buffer if possible
        try:
            idx = self._free_buffers.get_nowait()
            bp = self._buffer_pool[idx]
            np.copyto(bp['pos'], self.pos)
            np.copyto(bp['sizes'], self.sizes)
            np.copyto(bp['colors'], self.colors)
            np.copyto(bp['radii'], self.radius)
            try:
                self._ready_queue.put_nowait((idx, self._time))
            except queue.Full:
                try:
                    old = self._ready_queue.get_nowait()
                    if isinstance(old[0], int):
                        try:
                            self._free_buffers.put_nowait(old[0])
                        except Exception:
                            pass
                except Exception:
                    pass
                try:
                    self._ready_queue.put_nowait((idx, self._time))
                except Exception:
                    try:
                        self._free_buffers.put_nowait(idx)
                    except Exception:
                        pass
        except queue.Empty:
            try:
                self._ready_queue.put_nowait((self.pos.copy(), self.sizes.copy(), self.colors.copy(), self.radius.copy(), self._time))
            except Exception:
                pass

        # on-screen overlay for FPS / sim stats
        try:
            self._overlay = scene.visuals.Text('', color=(1, 1, 1, 1), font_size=12, anchor_x='left', anchor_y='top', parent=self.canvas.scene)
            try:
                self._overlay.set_gl_state('translucent', depth_test=False)
            except Exception:
                pass
            try:
                self._overlay.visible = True
            except Exception:
                pass
        except Exception:
            try:
                self._overlay = scene.visuals.Text('', color='white', font_size=12, anchor_x='left', anchor_y='top', parent=self.view.scene)
                try:
                    self._overlay.set_gl_state('translucent', depth_test=False)
                except Exception:
                    pass
                try:
                    self._overlay.visible = True
                except Exception:
                    pass
            except Exception:
                self._overlay = None

        # control to stop the simulation thread when canvas closes
        self._stop_event = threading.Event()
        try:
            self.canvas.events.close.connect(self._on_close)
        except Exception:
            pass

        # start update timer (render loop) on main thread
        # render framerate control: if render_fps is None or <=0, run as fast as possible
        self.render_fps = None if (render_fps is None or float(render_fps) <= 0) else float(render_fps)
        interval = 0 if self.render_fps is None else (1.0 / float(self.render_fps))
        self.timer = app.Timer(interval=interval, connect=self.update, start=True)

        # stats (exponential moving averages)
        self._stats_alpha = 0.05
        self.sim_step_time_ema = 0.0
        self.sim_batch_time_ema = 0.0
        self.publish_time_ema = 0.0
        self.render_frame_time_ema = 0.0
        self._sim_batches = 0
        self._render_frames = 0
        self._last_render_tick = None
        self._last_rendered_body_count = int(self.pos.shape[0])

        # optional periodic stats logging (runs on main thread)
        self._stats_timer = None
        if stats_log_interval is not None and stats_log_interval > 0:
            try:
                self._stats_timer = app.Timer(interval=float(stats_log_interval), connect=self._print_stats, start=True)
            except Exception:
                self._stats_timer = None

        # start background simulation thread (daemon)
        self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._sim_thread.start()

    def compute_total_energy_and_momentum(self, eps=0.0):
        N, d = self.pos.shape
        v2 = np.sum(self.vel * self.vel, axis=1)
        ke = 0.5 * float(np.sum(self.mass.astype(float) * v2))

        pe = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                r2 = 0.0
                for k in range(d):
                    diff = self.pos[i, k] - self.pos[j, k]
                    r2 += diff * diff
                r = math.sqrt(r2 + eps)
                pe -= float(self.G) * float(self.mass[i]) * float(self.mass[j]) / r

        momentum = np.sum(self.mass[:, np.newaxis] * self.vel, axis=0)
        return ke, pe, ke + pe, momentum

    def measure(self, eps=0.0):
        ke, pe, total, momentum = self.compute_total_energy_and_momentum(eps)
        self.energy_history.append((self._time, float(ke), float(pe), float(total)))
        self.momentum_history.append((self._time, momentum.copy()))
        self.time_history.append(self._time)

    def plot_history(self, save_dir=None, show=True):
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print('matplotlib not available; skipping plot generation.')
            return

        if len(self.energy_history) == 0:
            print('No history recorded; nothing to plot.')
            return

        times = np.array([t for t, ke, pe, tot in self.energy_history])
        ke = np.array([ke for t, ke, pe, tot in self.energy_history])
        pe = np.array([pe for t, ke, pe, tot in self.energy_history])
        total = np.array([tot for t, ke, pe, tot in self.energy_history])
        momentum = np.array([m for t, m in self.momentum_history])
        if momentum.ndim == 1:
            momentum = momentum.reshape(-1, 1)

        M0 = MSUN
        L0 = AU
        T0 = TUNIT
        conv_energy = (T0 * T0) / (M0 * (L0 * L0))
        conv_momentum = T0 / (M0 * L0)

        time_rel = times / T0
        ke_rel = ke * conv_energy
        pe_rel = pe * conv_energy
        total_rel = total * conv_energy
        momentum_rel = momentum * conv_momentum
        momentum_norm = np.linalg.norm(momentum_rel, axis=1)

        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
        ax = axes[0]
        ax.plot(time_rel, ke_rel, label='Kinetic Energy')
        ax.plot(time_rel, pe_rel, label='Potential Energy')
        ax.plot(time_rel, total_rel, label='Total Energy')
        ax.set_ylabel('Energy [M_sun * AU^2 / (yr/(2π))^2]')
        ax.grid(True)
        ax.legend()

        ax2 = axes[1]
        if momentum_rel.shape[1] >= 2:
            ax2.plot(time_rel, momentum_rel[:, 0], label='Px')
            ax2.plot(time_rel, momentum_rel[:, 1], label='Py')
        else:
            ax2.plot(time_rel, momentum_rel[:, 0], label='Px')
        ax2.plot(time_rel, momentum_norm, label='|P|', linestyle='--')
        ax2.set_ylabel('Momentum [M_sun * AU / (yr/(2π))]')
        ax2.set_xlabel('Time [yr/(2π)]')
        ax2.grid(True)
        ax2.legend()

        fig.tight_layout()
        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), 'plots')
        os.makedirs(save_dir, exist_ok=True)
        ts = __import__('datetime').datetime.now().strftime('%Y%m%d-%H%M%S')
        fname = os.path.join(save_dir, f'nbody_history_rel_{ts}.png')
        fig.savefig(fname)
        print(f'Saved history plot to {fname}')
        if show:
            plt.show()
        plt.close(fig)

    def compute_sizes(self):
        ref = self._get_radius_reference()
        self._radius_ref = ref
        try:
            # Avoid reading canvas/camera from the simulation thread.
            if self.marker_world_scaling and threading.current_thread() is threading.main_thread():
                world_sizes = self._compute_world_marker_sizes(self.radius)
                if world_sizes is not None:
                    self.sizes = world_sizes
                    return
        except Exception:
            pass

        sizes = self.marker_base * np.power((self.radius / ref), self.marker_scale_exponent)
        sizes = np.clip(sizes, 1.0, self.marker_max_size).astype(np.float32)
        self.sizes = sizes

    def _compute_world_marker_sizes(self, radii):
        if not (hasattr(self, 'canvas') and hasattr(self, 'view')):
            return None
        csize = getattr(self.canvas, 'size', None)
        crect = getattr(self.view.camera, 'rect', None)
        if csize is None or crect is None:
            return None

        cw, ch = float(csize[0]), float(csize[1])
        if hasattr(crect, 'width') and hasattr(crect, 'height'):
            ww, wh = float(crect.width), float(crect.height)
        else:
            ww, wh = float(crect[2]), float(crect[3])
        if cw <= 0.0 or ch <= 0.0 or ww <= 0.0 or wh <= 0.0:
            return None

        ppu_x = cw / ww
        ppu_y = ch / wh
        ppu = 0.5 * (ppu_x + ppu_y)

        radii_arr = np.asarray(radii, dtype=np.float64)
        diam = 2.0 * radii_arr * ppu * float(self.marker_world_scale)
        # In world-scaling mode, avoid an upper cap so visual geometry can stay physically faithful.
        return np.maximum(diam, 1.0).astype(np.float32)

    def _get_radius_reference(self):
        if self.radius.size == 0:
            return 1.0
        ref = float(np.median(self.radius))
        if ref <= 0.0:
            ref = float(np.mean(self.radius)) if self.radius.size > 0 else 1.0
        if ref <= 0.0:
            ref = 1.0
        return ref

    def resolve_collisions(self):
        if self.pos.shape[0] <= 1:
            return
        merged_any = False
        while True:
            N = self.pos.shape[0]
            if N <= 1:
                break
            diffs = self.pos[:, None, :] - self.pos[None, :, :]
            d2 = np.sum(diffs * diffs, axis=2)
            radii_sum = (self.radius[:, None] + self.radius[None, :])
            radii_thr2 = (radii_sum * self.collision_merge_margin) ** 2
            coll = np.triu(d2 <= radii_thr2, k=1)
            pairs = np.argwhere(coll)
            if pairs.shape[0] == 0:
                break
            i, j = pairs[0]
            mi = float(self.mass[i])
            mj = float(self.mass[j])
            new_mass = mi + mj
            new_pos = (mi * self.pos[i] + mj * self.pos[j]) / new_mass
            new_vel = (mi * self.vel[i] + mj * self.vel[j]) / new_mass

            if getattr(self, 'use_mass_radius', False):
                try:
                    new_radius = mass_to_radius(new_mass)
                except Exception:
                    new_radius = (new_mass / (4.0 / 3.0 * math.pi * max(self.density, 1.0))) ** (1.0 / 3.0)
            else:
                if self.density > 0.0:
                    new_radius = (new_mass / (4.0 / 3.0 * math.pi * self.density)) ** (1.0 / 3.0)
                else:
                    vol_i = (4.0 / 3.0) * math.pi * (self.radius[i] ** 3)
                    vol_j = (4.0 / 3.0) * math.pi * (self.radius[j] ** 3)
                    new_vol = vol_i + vol_j
                    new_radius = (new_vol * 3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)

            try:
                old_ri = float(self.radius[i])
                old_rj = float(self.radius[j])
                if new_radius < max(old_ri, old_rj):
                    new_radius = max(old_ri, old_rj)
            except Exception:
                pass

            self.pos[i] = new_pos
            self.vel[i] = new_vel
            self.mass[i] = new_mass
            self.radius[i] = new_radius
            try:
                self.colors[i] = temperature_to_rgba(mass_to_temperature(new_mass))[0].astype(np.float32)
            except Exception:
                self.colors[i] = np.ones(4, dtype=np.float32)

            self.pos = np.delete(self.pos, j, axis=0)
            self.vel = np.delete(self.vel, j, axis=0)
            self.mass = np.delete(self.mass, j, axis=0)
            self.radius = np.delete(self.radius, j, axis=0)
            self.colors = np.delete(self.colors, j, axis=0)

            self.acc = np.zeros_like(self.pos)
            self._tmp_pos = np.empty_like(self.pos)
            self._tmp_acc = np.empty_like(self.pos)

            compute_accelerations(self.pos, self.mass, self.radius, self.G, self.acc)

            self.compute_sizes()

            merged_any = True

        if merged_any:
            # Rendering runs on the main thread; simulation thread publishes snapshots only.
            if threading.current_thread() is threading.main_thread():
                try:
                    self.scatter.set_data(self.pos, size=self.sizes, face_color=self.colors)
                except Exception:
                    pass

    def _on_close(self, event=None):
        try:
            self._stop_event.set()
        except Exception:
            pass

    def _sim_loop(self):
        """Background simulation loop. Performs `physics_steps_per_frame` integrator
        steps and then publishes the newest snapshot to the internal queue.

        If `max_sim_steps_per_second` is set, the loop will throttle so it does not
        exceed that number of integrator steps per second.
        """
        target_step_time = None
        if self.max_sim_steps_per_second is not None and self.max_sim_steps_per_second > 0:
            target_step_time = 1.0 / float(self.max_sim_steps_per_second)

        while not self._stop_event.is_set():
            batch_start = time.perf_counter()

            # perform a batch of integrator steps and time them
            integrator_start = time.perf_counter()
            for _ in range(self.physics_steps_per_frame):
                yoshida_step(self.pos, self.vel, self.acc, self.dt, self.mass, self.radius, self._tmp_pos, self._tmp_acc, self.softening, self.G)
                if self.enable_collisions:
                    try:
                        self.resolve_collisions()
                    except Exception:
                        pass
                self._time += self.dt
            integrator_elapsed = time.perf_counter() - integrator_start

            # update EMAs for integrator times
            try:
                per_step = integrator_elapsed / float(self.physics_steps_per_frame)
            except Exception:
                per_step = integrator_elapsed
            alpha = getattr(self, '_stats_alpha', 0.05)
            if self._sim_batches == 0:
                self.sim_step_time_ema = per_step
                self.sim_batch_time_ema = integrator_elapsed
            else:
                self.sim_step_time_ema = alpha * per_step + (1.0 - alpha) * self.sim_step_time_ema
                self.sim_batch_time_ema = alpha * integrator_elapsed + (1.0 - alpha) * self.sim_batch_time_ema
            self._sim_batches += 1

            # Keep simulation batches close to a render-friendly budget.
            if self.adaptive_steps_per_frame and self.render_fps is not None and self._sim_batches >= self._next_adapt_batch:
                target_batch = 0.70 * (1.0 / float(self.render_fps))
                current = float(self.sim_batch_time_ema)
                if current > (1.25 * target_batch) and self.physics_steps_per_frame > self.min_physics_steps_per_frame:
                    self.physics_steps_per_frame = max(self.min_physics_steps_per_frame, self.physics_steps_per_frame - 1)
                elif current < (0.55 * target_batch) and self.physics_steps_per_frame < self.max_physics_steps_per_frame:
                    self.physics_steps_per_frame = min(self.max_physics_steps_per_frame, self.physics_steps_per_frame + 1)
                self._next_adapt_batch = self._sim_batches + self._adapt_every_batches

            # measure for history
            try:
                self.measure()
            except Exception:
                pass

            # publish newest state using a pooled buffer when possible
            publish_start = time.perf_counter()
            try:
                buf_idx = self._free_buffers.get_nowait()
                buf = self._buffer_pool[buf_idx]
                # reallocate buffers if shapes changed (e.g., after collisions)
                if buf['pos'].shape != self.pos.shape:
                    buf['pos'] = np.empty_like(self.pos)
                if buf['sizes'].shape != self.sizes.shape:
                    buf['sizes'] = np.empty_like(self.sizes)
                if buf['colors'].shape != self.colors.shape:
                    buf['colors'] = np.empty_like(self.colors)
                if buf['radii'].shape != self.radius.shape:
                    buf['radii'] = np.empty_like(self.radius)

                # copy into pooled buffers
                try:
                    np.copyto(buf['pos'], self.pos)
                    np.copyto(buf['sizes'], self.sizes)
                    np.copyto(buf['colors'], self.colors)
                    np.copyto(buf['radii'], self.radius)
                except Exception:
                    # fallback to full copy allocation if copyto fails
                    buf['pos'] = self.pos.copy()
                    buf['sizes'] = self.sizes.copy()
                    buf['colors'] = self.colors.copy()
                    buf['radii'] = self.radius.copy()

                # publish buffer index
                try:
                    self._ready_queue.put_nowait((buf_idx, self._time))
                except queue.Full:
                    # drop the older ready buffer and return it to free pool
                    try:
                        old = self._ready_queue.get_nowait()
                        if isinstance(old[0], int):
                            try:
                                self._free_buffers.put_nowait(old[0])
                            except Exception:
                                pass
                    except Exception:
                        pass
                    try:
                        self._ready_queue.put_nowait((buf_idx, self._time))
                    except Exception:
                        try:
                            self._free_buffers.put_nowait(buf_idx)
                        except Exception:
                            pass
            except queue.Empty:
                # no free pooled buffer; fall back to allocating copies
                try:
                    pos_copy = self.pos.copy()
                    sizes_copy = self.sizes.copy()
                    colors_copy = self.colors.copy()
                    radii_copy = self.radius.copy()
                except Exception:
                    pos_copy = self.pos.copy()
                    sizes_copy = getattr(self, 'sizes', np.ones((self.pos.shape[0],), dtype=np.float32)).copy()
                    colors_copy = getattr(self, 'colors', np.ones((self.pos.shape[0], 4), dtype=np.float32)).copy()
                    radii_copy = self.radius.copy()
                try:
                    self._ready_queue.put_nowait((pos_copy, sizes_copy, colors_copy, radii_copy, self._time))
                except queue.Full:
                    try:
                        old = self._ready_queue.get_nowait()
                        if isinstance(old[0], int):
                            try:
                                self._free_buffers.put_nowait(old[0])
                            except Exception:
                                pass
                    except Exception:
                        pass
                    try:
                        self._ready_queue.put_nowait((pos_copy, sizes_copy, colors_copy, radii_copy, self._time))
                    except Exception:
                        pass

            publish_elapsed = time.perf_counter() - publish_start

            # update publish-time EMA
            if self.publish_time_ema == 0.0:
                self.publish_time_ema = publish_elapsed
            else:
                self.publish_time_ema = alpha * publish_elapsed + (1.0 - alpha) * self.publish_time_ema

            # throttle to limit simulation speed if requested
            if target_step_time is not None:
                elapsed = time.perf_counter() - batch_start
                expected = self.physics_steps_per_frame * target_step_time
                if elapsed < expected:
                    time.sleep(expected - elapsed)
            else:
                # yield briefly to allow UI thread to run
                time.sleep(0)

    def update(self, event):
        # measure frame time and render latest snapshot if available
        render_start = time.perf_counter()
        frame_interval = None
        if self._last_render_tick is not None:
            frame_interval = max(0.0, render_start - self._last_render_tick)
        self._last_render_tick = render_start
        pos = None
        try:
            item = self._ready_queue.get_nowait()
        except queue.Empty:
            item = None

        if item is not None:
            # item can be (buf_idx, time) or (pos, sizes, colors, radii, time)
            if isinstance(item[0], int):
                buf_idx, t = item
                try:
                    buf = self._buffer_pool[buf_idx]
                    pos = buf['pos']
                    sizes = buf['sizes']
                    colors = buf['colors']
                    radii = buf['radii']

                    if self.marker_world_scaling:
                        try:
                            world_sizes = self._compute_world_marker_sizes(radii)
                            if world_sizes is not None and world_sizes.shape[0] == pos.shape[0]:
                                sizes = world_sizes
                        except Exception:
                            pass

                    try:
                        self.scatter.set_data(pos, size=sizes, face_color=colors)
                    except Exception:
                        try:
                            self.scatter.set_data(pos)
                        except Exception:
                            pass
                finally:
                    # return buffer to free pool
                    try:
                        self._free_buffers.put_nowait(buf_idx)
                    except Exception:
                        pass
            else:
                # raw arrays
                try:
                    pos, sizes, colors, radii, t = item

                    if self.marker_world_scaling:
                        try:
                            world_sizes = self._compute_world_marker_sizes(radii)
                            if world_sizes is not None and world_sizes.shape[0] == pos.shape[0]:
                                sizes = world_sizes
                        except Exception:
                            pass

                    try:
                        self.scatter.set_data(pos, size=sizes, face_color=colors)
                    except Exception:
                        try:
                            self.scatter.set_data(pos)
                        except Exception:
                            pass
                except Exception:
                    pass

        if pos is not None:
            try:
                self._last_rendered_body_count = int(pos.shape[0])
            except Exception:
                pass

        # update overlay with stats
        try:
            if self._overlay is not None:
                stats = self.get_stats()
                rframe = stats.get('render_frame_time_ema', 0.0)
                sstep = stats.get('sim_step_time_ema', 0.0)
                sbatch = stats.get('sim_batch_time_ema', 0.0)
                pub = stats.get('publish_time_ema', 0.0)
                numba_t = stats.get('numba_threads', None)
                n_bodies = stats.get('n_bodies', None)
                fps = (1.0 / rframe) if (rframe is not None and rframe > 1e-12) else 0.0
                txt = f"FPS: {fps:.1f}\nSim step: {sstep*1000:.3f} ms\nBatch: {sbatch*1000:.3f} ms\nPublish: {pub*1000:.3f} ms\nSPF: {self.physics_steps_per_frame}\nBodies: {n_bodies if n_bodies is not None else 'N/A'}\nThreads: {numba_t if numba_t is not None else 'auto'}"
                # position top-left in pixel coordinates (canvas size)
                try:
                    sz = getattr(self.canvas, 'size', None)
                    if sz is not None:
                        w, h = float(sz[0]), float(sz[1])
                        # set pos such that text appears near top-left
                        try:
                            self._overlay.pos = (10, h - 10)
                        except Exception:
                            # some Text visuals use set_pos
                            try:
                                self._overlay.set_pos((10, h - 10))
                            except Exception:
                                pass
                except Exception:
                    pass
                try:
                    self._overlay.text = txt
                except Exception:
                    try:
                        self._overlay.set_text(txt)
                    except Exception:
                        pass
        except Exception:
            pass

        # always request a redraw (keeps framerate consistent)
        self.canvas.update()

        # update render-time EMA
        render_elapsed = time.perf_counter() - render_start
        frame_time = frame_interval if (frame_interval is not None and frame_interval > 0.0) else render_elapsed
        alpha = getattr(self, '_stats_alpha', 0.05)
        if self._render_frames == 0:
            self.render_frame_time_ema = frame_time
        else:
            self.render_frame_time_ema = alpha * frame_time + (1.0 - alpha) * self.render_frame_time_ema
        self._render_frames += 1

    def _print_stats(self, ev=None):
        try:
            s = float(self.sim_step_time_ema)
            sb = float(self.sim_batch_time_ema)
            p = float(self.publish_time_ema)
            r = float(self.render_frame_time_ema)
            batches = int(self._sim_batches)
            frames = int(self._render_frames)
            steps_per_sec = None
            if s > 0.0:
                steps_per_sec = 1.0 / s
            steps_str = f"{steps_per_sec:.1f}" if steps_per_sec is not None else "N/A"
            print(f"[nbody] sim_step={s*1000:.3f}ms sim_batch={sb*1000:.3f}ms publish={p*1000:.3f}ms render={r*1000:.3f}ms batches={batches} frames={frames} steps/s={steps_str}")
        except Exception:
            pass

    def get_stats(self):
        return {
            'sim_step_time_ema': float(self.sim_step_time_ema),
            'sim_batch_time_ema': float(self.sim_batch_time_ema),
            'publish_time_ema': float(self.publish_time_ema),
            'render_frame_time_ema': float(self.render_frame_time_ema),
            'sim_batches': int(self._sim_batches),
            'render_frames': int(self._render_frames),
            'render_fps': self.render_fps,
            'numba_threads': getattr(self, 'numba_threads', None),
            'physics_steps_per_frame': int(self.physics_steps_per_frame),
            'n_bodies': int(getattr(self, '_last_rendered_body_count', self.pos.shape[0])),
            'window_backend': self.window_backend,
        }
