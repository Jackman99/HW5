''' This module defines the Fields class, which represents the velocity and pressure fields in a fluid simulation. '''
import numpy as np

class Fields:
    def __init__(self, grid):
        self.grid = grid
        shape = (grid.nx + 2, grid.ny + 2) # Including ghost cells
        self.u = np.zeros(shape) # Velocity in x-direction, stored on cell centers, size is nx+2, ny+2 because of ghost cells
        self.v = np.zeros(shape) # Velocity in y-direction, stored on cell centers, size is nx+2, ny+2 because of ghost cells
        self.p = np.zeros(shape) # Pressure, stored on cell centers, size is nx+2, ny+2 because of ghost cells
        self.U = np.zeros((grid.nx + 1, grid.ny + 1)) # Velocity in x-direction, stored on faces, size is nx + 1, ny
        self.V = np.zeros((grid.nx + 1, grid.ny + 1)) # Velocity in y-direction, stored on faces, size is nx, ny + 1
        # Previous time step values for U, V, P, u, v
        self.u_prev = np.zeros_like(self.u)
        self.v_prev = np.zeros_like(self.v)
        self.p_prev = np.zeros_like(self.p)
        self.U_prev = np.zeros_like(self.U)
        self.V_prev = np.zeros_like(self.V)