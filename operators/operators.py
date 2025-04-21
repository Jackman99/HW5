import numpy as np
import matplotlib.pyplot as plt
from Domain.grid import Grid
from numba import njit
from Domain.fields import Fields

@njit
def laplacian(field, dx, dy): 
    """Compute Laplacian of a scalar field using central differences.
    RETURNS SAME SIZE AS INPUT FIELD.
    dx, dy are the grid spacings in x and y directions.
    """
    lap = np.zeros_like(field)
    nx, ny = field.shape
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            lap[i, j] = (
                (field[i+1, j] - 2*field[i, j] + field[i-1, j]) / dx**2 +
                (field[i, j+1] - 2*field[i, j] + field[i, j-1]) / dy**2
            )
    return lap

@njit
def gradient(field, dx, dy, scheme):
    """
    Compute gradient (∂/∂x, ∂/∂y) of a scalar field with ghost cells.
    
    noncompact RETURNS SIZE SAME AS INPUT FIELD.
    compact RETURNS SIZE 1 SMALLER IN BOTH DIRECTIONS THAN INPUT.
    """
    nx, ny = field.shape

    if scheme == 'noncompact':
        grad_x = np.zeros_like(field)
        grad_y = np.zeros_like(field)
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                grad_x[i, j] = (field[i + 1, j] - field[i - 1, j]) / (2 * dx)
                grad_y[i, j] = (field[i, j + 1] - field[i, j - 1]) / (2 * dy)

    elif scheme == 'compact':
        grad_x = np.zeros((nx - 1, ny - 1))
        grad_y = np.zeros((nx - 1, ny - 1))
        for i in range(nx - 1):
            for j in range(ny - 1):
                grad_x[i, j] = (field[i + 1, j] - field[i, j]) / dx
                grad_y[i, j] = (field[i, j + 1] - field[i, j]) / dy

    else:
        raise ValueError("Unsupported scheme. Choose 'compact' or 'noncompact'.")

    return grad_x, grad_y

@njit
def divergence(u, v, dx, dy): 
    """Compute divergence ∇·(u, v) of a vector field.
    RETURNS SIZE 1 BIGGER IN BOTH DIRECTIONS THAN INPUT.
    """
    
    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape")
    
    nx, ny = u.shape
    div = np.zeros((nx+1, ny+1))  # one larger in both directions

    # Internal points (central difference)
    for i in range(1, nx):
        for j in range(1, ny):
            dudx = (u[i, j] - u[i-1, j]) / dx
            dvdy = (v[i, j] - v[i, j-1]) / dy
            div[i, j] = dudx + dvdy

    # Left and right boundaries (forward/backward diff in x)
    for j in range(1, ny):
        div[0, j] = (u[0, j] - 0.0) / dx + (v[0, j] - v[0, j-1]) / dy  # left
        div[nx, j] = (0.0 - u[nx-1, j]) / dx + (v[nx-1, j] - v[nx-1, j-1]) / dy  # right

    # Bottom and top boundaries (forward/backward diff in y)
    for i in range(1, nx):
        div[i, 0] = (u[i, 0] - u[i-1, 0]) / dx + (v[i, 0] - 0.0) / dy  # bottom
        div[i, ny] = (u[i, ny-1] - u[i-1, ny-1]) / dx + (0.0 - v[i, ny-1]) / dy  # top

    # Corners
    div[0, 0] = (u[0, 0] - 0.0) / dx + (v[0, 0] - 0.0) / dy
    div[0, ny] = (u[0, ny-1] - 0.0) / dx + (0.0 - v[0, ny-1]) / dy
    div[nx, 0] = (0.0 - u[nx-1, 0]) / dx + (v[nx-1, 0] - 0.0) / dy
    div[nx, ny] = (0.0 - u[nx-1, ny-1]) / dx + (0.0 - v[nx-1, ny-1]) / dy

    return div

@njit
def interpolate_center_to_face(field, axis): 
    """Linearly interpolate field along specified axis.
    RETURNS FIELD OF SIZE (n - 1, n - 1) than input field.
    axis = 0 for x-direction, axis = 1 for y-direction.
    """
    nx, ny = field.shape
    interpolated = np.zeros((nx - 1, ny - 1))
    if axis == 0:
        for i in range(nx - 1):
            for j in range(ny - 1):
                interpolated[i, j] = 0.5 * (field[i, j] + field[i+1, j])
        return interpolated
    
    elif axis == 1:
        for i in range(nx - 1):
            for j in range(ny - 1):
                interpolated[i, j] = 0.5 * (field[i, j] + field[i, j+1])
        return interpolated
    else:
        raise ValueError("Axis must be 0 or 1")

class Operators:
    def __init__(self, grid):
        self.nx = grid.nx # Number of grid points in x-direction
        self.ny = grid.ny # Number of grid points in y-direction
        self.Lx = grid.Lx # Length of the domain in x-direction
        self.Ly = grid.Ly # Length of the domain in y-direction
        self.dx = grid.dx # Grid spacing in x-direction
        self.dy = grid.dy # Grid spacing in y-direction
        self.Re = grid.Re # Reynolds number
        self.dt = grid.dt # Time step size

    def laplacian(self, field):
        return laplacian(field, self.dx, self.dy)

    def gradient(self, field, scheme):
        return gradient(field, self.dx, self.dy, scheme)

    def divergence(self, u, v):
        return divergence(u, v, self.dx, self.dy)

    def interpolate_center_to_face(self, field, axis): 
        return interpolate_center_to_face(field, axis)

    def gs_for_pressure(self, a_e, a_w, a_n, a_s, a_p, explicit_term, initial_guess, 
                    tol=1e-8, max_iter=1000, verbose=False):

        p_grid = np.copy(initial_guess)

        # Initialize residual tracking
        u_max_residual = np.zeros(max_iter)

        # Define numerical constants
        grid_size_x, grid_size_y = p_grid.shape
        ap_inverse = 1.0 / a_p

        # Apply boundary conditions initially
        p_grid[:, 0] = p_grid[:, 1] # Bottom (j=0): u = 0
        p_grid[:, -1] = p_grid[:, -2] # Top (j=-1): u = U_lid
        p_grid[0, :] = p_grid[1, :] # Left (i=0): u = 0
        p_grid[-1, :] =  p_grid[-2, :] # Right (i=-1): u = 0

        for iteration in range(max_iter):
            u_max_residual_value = 0.0

            # Gauss-Seidel iteration update
            for i in range(1, grid_size_x - 1):
                for j in range(1, grid_size_y - 1):

                    # Gauss-Seidel update with relaxation
                    p_grid[i, j] = ap_inverse * (
                        explicit_term[i, j] # explicit_u is 1 cell smaller than u_grid (no boundary cells)
                        + a_e * p_grid[i + 1, j] + a_w * p_grid[i - 1, j]
                        + a_n * p_grid[i, j + 1] + a_s * p_grid[i, j - 1]
                    ) 

                    # Compute residuals
                    residual_ij_u = np.abs(a_p * p_grid[i, j] - (
                        explicit_term[i, j]
                        + a_e * p_grid[i + 1, j] + a_w * p_grid[i - 1, j]
                        + a_n * p_grid[i, j + 1] + a_s * p_grid[i, j - 1]
                    ))

                    u_max_residual[iteration] = max(u_max_residual[iteration], residual_ij_u)

            # Store max residual for the iteration
            u_max_residual_value = u_max_residual[iteration]
            print(f'Iteration {iteration}: u_max_residual = {u_max_residual_value}')
            
            # Check for convergence
            if (u_max_residual_value < tol):
                print(f'Converged at iteration {iteration}')
                break

            # Apply boundary conditions at the end of each time step

            # p_grid[:, 0] = p_grid[:, 1] # Bottom (j=0): v = 0
            # p_grid[:, -1] = p_grid[:, -2] # Top (j=-1): v = 0
            # p_grid[0, :] = p_grid[1, :] # Left (i=0): v = 0
            # p_grid[-1, :] = p_grid[-2, :] # Right (i=-1): v = 0

        return p_grid, u_max_residual, iteration + 1

    def gs_for_velocity(self, a_e, a_w, a_n, a_s, a_p, explicit_term, initial_guess, 
                        tol=1e-8, max_iter=1000, verbose=False):
        # initial guess is a tuple of (u, v) velocity fields
        # Explicit term is a tuple of (explicit_u, explicit_v) terms
        # Initialize velocity fields
        u_grid = np.copy(initial_guess[0])
        v_grid = np.copy(initial_guess[1])

        # Initialize residual tracking
        u_max_residual = np.zeros(max_iter)
        v_max_residual = np.zeros(max_iter)

        # Define numerical constants
        grid_size_x, grid_size_y = u_grid.shape
        ap_inverse = 1.0 / a_p

        # Apply boundary conditions initially
        u_grid[:, 0] = -u_grid[:, 1] # Bottom (j=0): u = 0
        u_grid[:, -1] = 2 * 1.0 -u_grid[:, -2] # Top (j=-1): u = U_lid
        u_grid[0, :] = -u_grid[1, :] # Left (i=0): u = 0
        u_grid[-1, :] =  - u_grid[-2, :] # Right (i=-1): u = 0

        v_grid[:, 0] = -v_grid[:, 1] # Bottom (j=0): v = 0
        v_grid[:, -1] = -v_grid[:, -2] # Top (j=-1): v = 0
        v_grid[0, :] = -v_grid[1, :] # Left (i=0): v = 0
        v_grid[-1, :] = -v_grid[-2, :] # Right (i=-1): v = 0

        for iteration in range(max_iter):
            u_max_residual_value = 0.0
            v_max_residual_value = 0.0

            # Gauss-Seidel iteration update
            for i in range(1, grid_size_x - 1):
                for j in range(1, grid_size_y - 1):

                    # Gauss-Seidel update with relaxation
                    u_grid[i, j] = ap_inverse * (
                        explicit_term[0][i, j] # explicit_u is 1 cell smaller than u_grid (no boundary cells)
                        + a_e * u_grid[i + 1, j] + a_w * u_grid[i - 1, j]
                        + a_n * u_grid[i, j + 1] + a_s * u_grid[i, j - 1]
                    ) 

                    v_grid[i, j] = ap_inverse * (
                        explicit_term[1][i, j]
                        + a_e * v_grid[i + 1, j] + a_w * v_grid[i - 1, j]
                        + a_n * v_grid[i, j + 1] + a_s * v_grid[i, j - 1]
                    )

                    # Compute residuals
                    residual_ij_u = np.abs(a_p * u_grid[i, j] - (
                        explicit_term[0][i, j]
                        + a_e * u_grid[i + 1, j] + a_w * u_grid[i - 1, j]
                        + a_n * u_grid[i, j + 1] + a_s * u_grid[i, j - 1]
                    ))
                    residual_ij_v = np.abs(a_p * v_grid[i, j] - (
                        explicit_term[1][i, j]
                        + a_e * v_grid[i + 1, j] + a_w * v_grid[i - 1, j]
                        + a_n * v_grid[i, j + 1] + a_s * v_grid[i, j - 1]
                    ))
                    u_max_residual[iteration] = max(u_max_residual[iteration], residual_ij_u)
                    v_max_residual[iteration] = max(v_max_residual[iteration], residual_ij_v)

            # Store max residual for the iteration
            u_max_residual_value = u_max_residual[iteration]
            v_max_residual_value = v_max_residual[iteration]
            print(f'Iteration {iteration}: u_max_residual = {u_max_residual_value}, v_max_residual = {v_max_residual_value}')
            
            # Check for convergence
            if (u_max_residual_value < tol) and (v_max_residual_value < tol):
                print(f'Converged at iteration {iteration}')
                break

            # Apply boundary conditions at the end of each time step
            u_grid[:, 0] = -u_grid[:, 1] # Bottom (j=0): u = 0
            u_grid[:, -1] = -u_grid[:, -2] # Top (j=-1): u = U_lid
            u_grid[0, :] = -u_grid[1, :] # Left (i=0): u = 0
            u_grid[-1, :] = 2 * 1.0 - u_grid[-2, :] # Right (i=-1): u = 0

            v_grid[:, 0] = -v_grid[:, 1] # Bottom (j=0): v = 0
            v_grid[:, -1] = -v_grid[:, -2] # Top (j=-1): v = 0
            v_grid[0, :] = -v_grid[1, :] # Left (i=0): v = 0
            v_grid[-1, :] = -v_grid[-2, :] # Right (i=-1): v = 0

        return u_grid, v_grid, u_max_residual, v_max_residual, iteration + 1

    def jacobi_solver(self, ae, aw, an, a_s, ap, explicit_term, initial_guess,
                      omega=1.0, tol=1e-6, max_iter=1000, verbose=False):
        phi_old = initial_guess.copy()
        phi_new = phi_old.copy()
        nx, ny = explicit_term.shape
        residuals = []

        for iteration in range(max_iter):
            max_residual = 0.0
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    rhs = (explicit_term[i-1, j-1] +
                        ae * phi_old[i+1, j] +
                        aw * phi_old[i-1, j] +
                        an * phi_old[i, j+1] +
                        a_s * phi_old[i, j-1])

                    phi_val = rhs / ap
                    phi_new[i, j] = (1 - omega) * phi_old[i, j] + omega * phi_val
                    residual = np.abs(ap * phi_new[i, j] - rhs)
                    max_residual = max(max_residual, residual)

            residuals.append(max_residual)
            if verbose:
                print(f"Iteration {iteration}: max residual = {max_residual:.2e}")
            if max_residual < tol:
                break
            
            # Neumann BC for Pressure
            phi_new[0, :] = phi_new[1, :]
            phi_new[-1, :] = phi_new[-2, :]
            phi_new[:, 0] = phi_new[:, 1]
            phi_new[:, -1] = phi_new[:, -2]

            phi_old = phi_new.copy()

        return phi_new, residuals, iteration + 1


