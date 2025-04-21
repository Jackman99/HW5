''' This module defines the parameters of a grid for 2D simulations'''
import numpy as np
import matplotlib.pyplot as plt

class Grid:
    '''This class represents a 2D grid for fluid simulations.
    It defines the grid size, cell dimensions, and provides methods for grid operations.'''

    def __init__(self, nx, ny, Lx, Ly, Re, dt):
        self.nx = nx # Number of grid points in x-direction
        self.ny = ny # Number of grid points in y-direction
        self.Lx = Lx # Length of the domain in x-direction
        self.Ly = Ly # Length of the domain in y-direction
        self.dx = Lx / nx # Grid spacing in x-direction
        self.dy = Ly / ny # Grid spacing in y-direction
        self.Re = Re # Reynolds number
        self.dt = dt # Time step size
