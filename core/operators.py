import numpy as np
import matplotlib.pyplot as plt
from core.grid import Grid
from numba import njit
from core.fields import Fields

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