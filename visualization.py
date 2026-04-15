import numpy as np
import math
from vispy import app, scene

from physics import MSUN, AU, TUNIT, mass_to_radius, mass_to_temperature, compute_accelerations, yoshida_step
from utils import masses_to_rgba, temperature_to_rgba


class NBodySimulation:
    def __init__(self, positions, velocities, masses, radii, G, dt, physics_steps_per_frame=1, marker_base=8.0, marker_scale_exponent=0.5, marker_max_size=80.0, marker_world_scaling=False, marker_world_scale=1.0, enable_collisions=True, density=5500.0, softening=0.1, collision_merge_margin=1.0, use_mass_radius=False, show_canvas=True):
        self.pos = positions.copy()
        self.vel = velocities.copy()
        self.mass = masses
        self.radius = radii

        total_mass = np.sum(self.mass)
        com_pos = np.sum(self.mass[:, np.newaxis] * self.pos, axis=0) / total_mass
        self.pos -= com_pos
        com_velocity = np.sum(self.mass[:, np.newaxis] * self.vel, axis=0) / total_mass
        self.vel -= com_velocity

        self.G = G
        self.dt = dt
        self.enable_collisions = bool(enable_collisions)
        self.density = float(density)
        self.use_mass_radius = bool(use_mass_radius)
        self.softening = float(softening)
        self.collision_merge_margin = float(collision_merge_margin)
        self.marker_base = float(marker_base)
        self.marker_scale_exponent = float(marker_scale_exponent)
        self.marker_max_size = float(marker_max_size)
        self.marker_world_scaling = bool(marker_world_scaling)
        self.marker_world_scale = float(marker_world_scale)

        self.physics_steps_per_frame = int(max(1, physics_steps_per_frame))

        self.acc = np.zeros_like(self.pos)
        self._tmp_pos = np.empty_like(self.pos)
        self._tmp_acc = np.empty_like(self.pos)

        compute_accelerations(self.pos, self.mass, self.radius, self.G, self.acc, 1e-30, 64, self.softening)

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

        ref_r = float(np.mean(self.radius)) if self.radius.size > 0 else 1.0
        if ref_r <= 0.0:
            ref_r = 1.0
        self._radius_ref = ref_r
        sizes = self.marker_base * np.power((self.radius / self._radius_ref), self.marker_scale_exponent)
        sizes = np.clip(sizes, 1.0, self.marker_max_size).astype(np.float32)
        self.sizes = sizes
        try:
            self.colors = masses_to_rgba(self.mass)
        except Exception:
            self.colors = np.ones((len(self.mass), 4), dtype=np.float32)

        self.canvas = scene.SceneCanvas(keys='interactive', show=show_canvas)
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

        self.timer = app.Timer(interval=0, connect=self.update, start=True)

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
        ref = getattr(self, '_radius_ref', float(np.mean(self.radius)) or 1.0)
        try:
            if self.marker_world_scaling and hasattr(self, 'canvas') and hasattr(self.view):
                csize = getattr(self.canvas, 'size', None)
                crect = getattr(self.view.camera, 'rect', None)
                if csize is not None and crect is not None:
                    cw, ch = float(csize[0]), float(csize[1])
                    ww, wh = float(crect[2]), float(crect[3])
                    if ww > 0 and wh > 0:
                        ppu_x = cw / ww
                        ppu_y = ch / wh
                        ppu = 0.5 * (ppu_x + ppu_y)
                        diam = 2.0 * self.radius * ppu * float(self.marker_world_scale)
                        sizes = np.clip(diam, 1.0, self.marker_max_size).astype(np.float32)
                        self.sizes = sizes
                        return
        except Exception:
            pass

        sizes = self.marker_base * np.power((self.radius / ref), self.marker_scale_exponent)
        sizes = np.clip(sizes, 1.0, self.marker_max_size).astype(np.float32)
        self.sizes = sizes

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

            ref = getattr(self, '_radius_ref', float(np.mean(self.radius)) or 1.0)
            self.sizes = self.marker_base * np.power((self.radius / ref), self.marker_scale_exponent)
            self.sizes = np.clip(self.sizes, 1.0, self.marker_max_size).astype(np.float32)

            merged_any = True

        if merged_any:
            try:
                self.scatter.set_data(self.pos, size=self.sizes, face_color=self.colors)
            except Exception:
                pass

    def update(self, event):
        for _ in range(self.physics_steps_per_frame):
            yoshida_step(self.pos, self.vel, self.acc, self.dt, self.mass, self.radius, self._tmp_pos, self._tmp_acc, self.softening, self.G)
            if self.enable_collisions:
                try:
                    self.resolve_collisions()
                except Exception:
                    pass
        self._time += self.physics_steps_per_frame * self.dt
        self.measure()

        self.scatter.set_data(self.pos, size=self.sizes, face_color=self.colors)
        self.canvas.update()
