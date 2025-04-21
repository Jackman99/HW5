import numpy as np

class PoissonSolver:
    def __init__(self, grid):
        self.dx = grid.dx
        self.dy = grid.dy

    def solve(self, rhs, max_iter=10000, tol=1e-5):
        P = np.zeros_like(rhs)
        for _ in range(max_iter):
            P_new = np.copy(P)
            P_new[1:-1, 1:-1] = 0.25 * (
                P[2:, 1:-1] + P[:-2, 1:-1] + P[1:-1, 2:] + P[1:-1, :-2] -
                self.dx * self.dy * rhs[1:-1, 1:-1]
            )
            if np.linalg.norm(P_new - P, ord=np.inf) < tol:
                break
            P = P_new
        return P
    
#     # --- poisson_solver.py ---
# import cupy as cp

# class PoissonSolver_GPU:
#     def __init__(self, grid):
#         self.dx = grid.dx
#         self.dy = grid.dy

#     def solve(self, rhs, max_iter=10000, tol=1e-5):
#         rhs_gpu = cp.asarray(rhs)
#         P = cp.zeros_like(rhs_gpu)
#         for _ in range(max_iter):
#             P_new = cp.copy(P)
#             P_new[1:-1, 1:-1] = 0.25 * (
#                 P[2:, 1:-1] + P[:-2, 1:-1] + P[1:-1, 2:] + P[1:-1, :-2] -
#                 self.dx * self.dy * rhs_gpu[1:-1, 1:-1]
#             )
#             if cp.linalg.norm(P_new - P, ord=cp.inf) < tol:
#                 break
#             P = P_new
#         return cp.asnumpy(P)