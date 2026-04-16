# =======================
# URUCHOMIENIE (prosty runner)
# =======================

from physics import random_masses, create_rotating_disk, mass_to_radius, TUNIT, AU
from visualization import NBodySimulation
from vispy import app
import numpy as np


def main():
    N = 80
    timestep = 0.1  # hours
    collision_merge_factor = 5.0
    # Display-only radius multiplier (does not affect physics or collision logic).
    display_radius_multiplier = 5.0
    # Keep small physics batches so render updates stay responsive.
    steps_per_frame = 2

    masses = random_masses(N)
    positions, velocities = create_rotating_disk(N, masses, r_min=1e10/AU, r_max=5e11/AU, v_scale=0.8, inward_fraction=0.02, clockwise=True, seed=(np.random.randint(0, 1e6)))
    radii = mass_to_radius(masses)

    sim = NBodySimulation(
        positions=positions,
        velocities=velocities*0.01,
        masses=masses,
        radii=radii,
        dt=timestep * 3600.0 / TUNIT,
        physics_steps_per_frame=steps_per_frame,
        render_fps=120,
        max_sim_steps_per_second=240,
        leave_one_core=True,
        stats_log_interval=1.0,
        collision_merge_margin=collision_merge_factor,
        use_mass_radius=True,
        marker_scale_exponent=1.0,
        # Render diameters in true world-space units (radius vs distance is geometrically consistent).
        marker_world_scaling=True,
        marker_world_scale=display_radius_multiplier,
        adaptive_steps_per_frame=True,
        min_physics_steps_per_frame=1,
        max_physics_steps_per_frame=24,
        G=1,
    )

    try:
        app.run()
    finally:
        # Always save a history plot when the simulation loop exits.
        try:
            sim.plot_history(save_dir='plots', show=False)
        except Exception as exc:
            print(f"[nbody] Failed to save history plot: {exc}")


if __name__ == "__main__":
    main()
