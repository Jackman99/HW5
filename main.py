import numpy as np
import os
from Domain.grid import Grid
from Domain.fields import Fields
from operators import Operators
from time_integrators import TimeIntegrator

# Parameters to sweep
reynolds_numbers = [100, 1000]
grid_sizes = [128]
Lx, Ly = 1.0, 1.0
dt = 0.005
num_steps = 5000

# Create output directory
output_dir = "nse_results"
os.makedirs(output_dir, exist_ok=True)

# Loop over all combinations of Re and grid size
for Re in reynolds_numbers:
    for N in grid_sizes:
        print(f"Running simulation for Re = {Re}, Grid = {N}x{N}")

        # Initialize components
        grid = Grid(N, N, Lx, Ly, Re, dt)
        fields = Fields(grid)
        ops = Operators(grid)
        integrator = TimeIntegrator(fields, ops)

        # Time integration loop
        for step in range(num_steps):
            max_div = integrator.advance_one_step()
            if step % 100 == 0:
                print(f"  Step {step}, Max Divergence: {max_div:.2e}")

        # Crop ghost cells for output
        u = fields.u[1:-1, 1:-1]
        v = fields.v[1:-1, 1:-1]
        p = fields.p[1:-1, 1:-1]

        # Save results
        filename = f"Re{int(Re)}_N{N}.npz"
        filepath = os.path.join(output_dir, filename)
        np.savez_compressed(filepath, u=u, v=v, p=p, Re=Re, N=N, dt=dt)
        print(f"Saved results to {filepath}")
