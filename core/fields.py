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

    def apply_boundary_conditions(self, bc_type, phi=None, U_lid=1.0):
        """
        Apply boundary conditions to a field. If `phi` is None, applies to self.u, self.v, or self.P based on `bc_type`.
        If `phi` is provided, applies the BC to that array directly.

        Parameters:
        - bc_type: str, one of {'u', 'v', 'p'}
        - phi: np.ndarray or None
        """
        # Determine which field to apply BCs to
        if phi is None:
            if bc_type == 'u':
                phi = self.u
            elif bc_type == 'v':
                phi = self.v
            elif bc_type == 'p':
                phi = self.P
            else:
                raise ValueError("Invalid boundary condition type. Choose 'U', 'V', or 'P'.")

        # Apply BCs based on type
        if bc_type == 'u':
            phi[:, 0] = -phi[:, 1] # Bottom (j=0): u = 0
            phi[:, -1] = -phi[:, -2] # Top (j=-1): u = U_lid
            phi[0, :] = -phi[1, :] # Left (i=0): u = 0
            phi[-1, :] = 2 * U_lid - phi[-2, :] # Right (i=-1): u = 0

        elif bc_type == 'v':
            phi[:, 0] = -phi[:, 1] # Bottom (j=0): v = 0
            phi[:, -1] = -phi[:, -2] # Top (j=-1): v = 0
            phi[0, :] = -phi[1, :] # Left (i=0): v = 0
            phi[-1, :] = -phi[-2, :] # Right (i=-1): v = 0

        elif bc_type == 'p':
            phi[0, :] = phi[1, :]          # Bottom
            phi[-1, :] = phi[-2, :]        # Top
            phi[:, 0] = phi[:, 1]          # Left
            phi[:, -1] = phi[:, -2]        # Right

        else:
            raise ValueError("Invalid boundary condition type. Choose 'u', 'v', or 'p'.")