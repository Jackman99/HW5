import numpy as np

class LinearSolvers:
    def __init__(self, grid):
        self.dx = grid.dx
        self.dy = grid.dy
        self.dt = grid.dt

    def gauss_seidel_velocity(self, a_e, a_w, a_n, a_s, a_p, explicit_term, initial_guess, bc_func,
                              tol=1e-8, max_iter=1000):
        u, v = np.copy(initial_guess[0]), np.copy(initial_guess[1])
        nx, ny = u.shape
        ap_inv = 1.0 / a_p
        u_res, v_res = [], []

        for it in range(max_iter):
            max_ures, max_vres = 0.0, 0.0
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    u_new = ap_inv * (explicit_term[0][i, j] +
                                      a_e * u[i+1, j] + a_w * u[i-1, j] +
                                      a_n * u[i, j+1] + a_s * u[i, j-1])
                    v_new = ap_inv * (explicit_term[1][i, j] +
                                      a_e * v[i+1, j] + a_w * v[i-1, j] +
                                      a_n * v[i, j+1] + a_s * v[i, j-1])
                    max_ures = max(max_ures, abs(u_new - u[i, j]))
                    max_vres = max(max_vres, abs(v_new - v[i, j]))
                    u[i, j], v[i, j] = u_new, v_new

            bc_func(u, v)
            u_res.append(max_ures)
            v_res.append(max_vres)
            if max_ures < tol and max_vres < tol:
                break
        return u, v, u_res, v_res, it+1

    def gauss_seidel_pressure(self, a_e, a_w, a_n, a_s, a_p, rhs, initial_guess, bc_func,
                              tol=1e-8, max_iter=1000):
        p = np.copy(initial_guess)
        nx, ny = p.shape
        ap_inv = 1.0 / a_p
        res = []

        for it in range(max_iter):
            max_res = 0.0
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    new_val = ap_inv * (
                        rhs[i, j] +
                        a_e * p[i+1, j] + a_w * p[i-1, j] +
                        a_n * p[i, j+1] + a_s * p[i, j-1]
                    )
                    max_res = max(max_res, abs(new_val - p[i, j]))
                    p[i, j] = new_val
            bc_func(p)
            res.append(max_res)
            if max_res < tol:
                break
        return p, res, it + 1

    def jacobi_scalar(self, a_e, a_w, a_n, a_s, a_p, rhs, initial_guess, bc_func,
                      omega=1.0, tol=1e-6, max_iter=1000):
        phi_old = initial_guess.copy()
        phi_new = phi_old.copy()
        nx, ny = phi_old.shape
        residuals = []

        for it in range(max_iter):
            max_res = 0.0
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    val = (
                        rhs[i, j] +
                        a_e * phi_old[i+1, j] + a_w * phi_old[i-1, j] +
                        a_n * phi_old[i, j+1] + a_s * phi_old[i, j-1]
                    )
                    phi_new[i, j] = (1 - omega) * phi_old[i, j] + omega * val / a_p
                    max_res = max(max_res, abs(phi_new[i, j] - phi_old[i, j]))
            bc_func(phi_new)
            phi_old[:, :] = phi_new
            residuals.append(max_res)
            if max_res < tol:
                break
        return phi_new, residuals, it + 1
