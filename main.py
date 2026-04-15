# =======================
# URUCHOMIENIE (prosty runner)
# =======================

from physics import random_masses, create_rotating_disk, mass_to_radius, TUNIT, AU
from visualization import NBodySimulation
from vispy import app
import numpy as np


def main():
    N = 100
    steps_per_frame = 25
    timestep = 0.01  # hours
    collision_merge_factor = 2

    masses = random_masses(N)
    positions, velocities = create_rotating_disk(N, masses, r_min=1e10/AU, r_max=5e11/AU, v_scale=0.8, inward_fraction=0.02, clockwise=True, seed=(np.random.randint(0, 1e6)))
    radii = mass_to_radius(masses)

    sim = NBodySimulation(
        positions=positions,
        velocities=velocities,
        masses=masses,
        radii=radii,
        dt=timestep * 3600.0 / TUNIT,
        physics_steps_per_frame=steps_per_frame,
        collision_merge_margin=collision_merge_factor,
        use_mass_radius=True,
        G=1,
    )

    app.run()


if __name__ == "__main__":
    main()
