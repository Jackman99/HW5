import numpy as np
import matplotlib.pyplot as plt
from Fields import Grid, Operators
# ======================= UNIT TESTS ===========================
if __name__ == "__main__":
    nx, ny, Lx, Ly = 100, 100, 1.0, 1.0
    grid = Grid(nx, ny, Lx, Ly, Re=1.0, dt=0.01)
    ops = Operators(grid)

    x = np.linspace(0, 1, nx + 2)
    y = np.linspace(0, 1, ny + 2)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # === Test 1: Trigonometric Laplacian and Gradient ===
    phi = np.sin(np.pi * X) * np.sin(np.pi * Y)
    lap_exact = -2 * np.pi**2 * phi
    lap_num = ops.laplacian(phi)
    grad_x_exact = np.pi * np.cos(np.pi * X) * np.sin(np.pi * Y)
    grad_y_exact = np.pi * np.sin(np.pi * X) * np.cos(np.pi * Y)
    grad_x, grad_y = ops.gradient(phi)

    plt.figure(); plt.title("Laplacian Error: sin(pi x) sin(pi y)")
    plt.imshow((lap_num - lap_exact).T[1:-1,1:-1], origin='lower'); plt.colorbar()

    plt.figure(); plt.title("Gradient X Error")
    plt.imshow((grad_x - grad_x_exact).T[1:-1,1:-1], origin='lower'); plt.colorbar()
    plt.figure(); plt.title("Gradient Y Error")
    plt.imshow((grad_y - grad_y_exact).T[1:-1,1:-1], origin='lower'); plt.colorbar()

    # === Test 2: Polynomial Laplacian and Gradient ===
    phi = X**2 + Y**2
    grad_x_exact = 2 * X
    grad_y_exact = 2 * Y
    grad_x, grad_y = ops.gradient(phi)
    lap_exact = np.full_like(phi, 4)
    lap_num = ops.laplacian(phi)

    plt.figure(); plt.title("Gradient X Error: x^2 + y^2")
    plt.imshow((grad_x - grad_x_exact).T[1:-1,1:-1], origin='lower'); plt.colorbar()
    plt.figure(); plt.title("Gradient Y Error: x^2 + y^2")
    plt.imshow((grad_y - grad_y_exact).T[1:-1,1:-1], origin='lower'); plt.colorbar()
    plt.figure(); plt.title("Laplacian Error: x^2 + y^2")
    plt.imshow((lap_num - lap_exact).T[1:-1,1:-1], origin='lower'); plt.colorbar()

    # === Test 3: Interpolation Accuracy ===
    field = np.sin(2 * np.pi * X) + np.cos(2 * np.pi * Y)

    interp_x = ops.interpolate(field, axis=0)
    # find true interpolation values with same grid size
    x_mid = 0.5 * (x[:-1] + x[1:])
    X_mid, Y_fixed = np.meshgrid(x_mid, y, indexing='ij')
    true_interp_x = np.sin(2 * np.pi * X_mid) + np.cos(2 * np.pi * Y_fixed) #TODO
    error_x = interp_x - true_interp_x

    interp_y = ops.interpolate(field, axis=1)
    y_mid = 0.5 * (y[:-1] + y[1:])
    X_fixed, Y_mid = np.meshgrid(x, y_mid, indexing='ij')
    true_interp_y = np.sin(2 * np.pi * X_fixed) + np.cos(2 * np.pi * Y_mid)
    error_y = interp_y - true_interp_y

    plt.figure(); plt.title("Interpolation Error (x-dir)")
    plt.imshow(error_x.T, origin="lower"); plt.colorbar()
    plt.figure(); plt.title("Interpolation Error (y-dir)")
    plt.imshow(error_y.T, origin="lower"); plt.colorbar()

    print("L2 error (x-dir):", np.sqrt(np.mean(error_x**2)))
    print("L2 error (y-dir):", np.sqrt(np.mean(error_y**2)))

    # === Divergence-Free Test ===
    u = np.sin(np.pi * Y)
    v = np.sin(np.pi * X)
    div = ops.divergence(u, v)
    plt.figure(); plt.title("Divergence Test (Should be ~0)")
    plt.imshow(div.T[1:-1, 1:-1], origin='lower'); plt.colorbar()

    # === GS & Jacobi Solver Tests ===
    f = np.ones((nx, ny)) * 4 * np.pi**2 * np.sin(np.pi * x[1:-1, None]) * np.sin(np.pi * y[None, 1:-1])
    u_exact = np.sin(np.pi * x[:, None]) * np.sin(np.pi * y[None, :])
    guess = np.zeros((nx + 2, ny + 2))

    ae = aw = an = a_s = 1.0 / grid.dx**2
    ap = -2.0 * (1.0 / grid.dx**2 + 1.0 / grid.dy**2)

    phi_gs, res_gs, _ = ops.gauss_seidel_solver(ae, aw, an, a_s, ap, f, guess, omega=1.2)
    phi_jac, res_jac, _ = ops.jacobi_solver(ae, aw, an, a_s, ap, f, guess, omega=0.8)

    plt.figure(); plt.title("GS Solution")
    plt.imshow(phi_gs.T[1:-1, 1:-1], origin='lower'); plt.colorbar()
    plt.figure(); plt.title("Jacobi Solution")
    plt.imshow(phi_jac.T[1:-1, 1:-1], origin='lower'); plt.colorbar()
    plt.figure(); plt.title("GS Residuals")
    plt.semilogy(res_gs)
    plt.figure(); plt.title("Jacobi Residuals")
    plt.semilogy(res_jac)

    plt.show()