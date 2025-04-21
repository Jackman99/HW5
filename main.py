import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.grid import Grid
from core.fields import Fields
from core.operators import Operators
from core.integrator import TimeIntegrator
import numpy as np


# Config
REYNOLDS_NUMBERS = [100, 1000]
GRID_SIZES = [32]
LX, LY = 1.0, 1.0
DT = 0.005
NUM_STEPS = 5000
OUTPUT_DIR = "nse_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for Re in REYNOLDS_NUMBERS:
    for N in GRID_SIZES:
        print(f"Running simulation for Re = {Re}, Grid = {N}x{N}")

        grid = Grid(N, N, LX, LY, Re, DT)
        fields = Fields(grid)
        ops = Operators(grid)
        integrator = TimeIntegrator(fields, ops)

        for step in range(NUM_STEPS):
            max_div = integrator.advance()
            if step % 100 == 0:
                print(f"  Step {step}, Max Divergence: {max_div:.2e}")

        u = fields.u[1:-1, 1:-1]
        v = fields.v[1:-1, 1:-1]
        p = fields.p[1:-1, 1:-1]

        filepath = os.path.join(OUTPUT_DIR, f"Re{int(Re)}_N{N}.npz")
        np.savez_compressed(filepath, u=u, v=v, p=p, Re=Re, N=N, dt=DT)
        print(f"Saved results to {filepath}")
