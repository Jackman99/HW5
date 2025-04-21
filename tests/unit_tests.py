import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import numpy as np
from core.grid import Grid
from core.fields import Fields
from core.operators import Operators
from utils.boundary_conditions import apply_pressure_bc, apply_velocity_bc
from core.solvers import LinearSolvers


class TestOperators(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(10, 10, 1.0, 1.0, Re=100, dt=0.01)
        self.ops = Operators(self.grid)

    def test_laplacian_polynomial(self):
        nx, ny = self.grid.nx + 2, self.grid.ny + 2
        dx, dy = self.grid.dx, self.grid.dy
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        phi = X**2 + Y**2
        lap_phi = self.ops.laplacian(phi)
        expected = np.full_like(phi, 4.0)
        error = np.max(np.abs(lap_phi[1:-1, 1:-1] - expected[1:-1, 1:-1]))
        self.assertLess(error, 1e-1)

    def test_gradient_constant(self):
        phi = np.ones((12, 12)) * 5.0
        grad_x, grad_y = self.ops.gradient(phi, scheme='noncompact')
        self.assertTrue(np.allclose(grad_x[1:-1, 1:-1], 0))
        self.assertTrue(np.allclose(grad_y[1:-1, 1:-1], 0))

    def test_divergence_zero(self):
        u = np.zeros((12, 12))
        v = np.zeros((12, 12))
        div = self.ops.divergence(u, v)
        self.assertTrue(np.allclose(div[1:-1, 1:-1], 0))

class TestSolvers(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(10, 10, 1.0, 1.0, Re=100, dt=0.01)
        self.solvers = LinearSolvers(self.grid)

    def test_gauss_seidel_pressure_convergence(self):
        nx, ny = self.grid.nx + 2, self.grid.ny + 2
        rhs = np.ones((nx, ny))
        p0 = np.zeros_like(rhs)
        a = -1 / self.grid.dx**2
        ap = -4 * a
        p, res, _ = self.solvers.gauss_seidel_pressure(a, a, a, a, ap, rhs, p0, apply_pressure_bc)
        self.assertLess(res[-1], 1e-3)

    def test_velocity_bc_application(self):
        u = np.zeros((12, 12))
        v = np.zeros((12, 12))
        apply_velocity_bc(u, v, lid_velocity=1.0)
        self.assertTrue(np.allclose(u[:, 0], -u[:, 1]))
        self.assertTrue(np.allclose(v[:, -1], -v[:, -2]))

if __name__ == '__main__':
    unittest.main()
