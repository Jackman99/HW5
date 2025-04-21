import numpy as np

class TimeIntegrator:
    def __init__(self, fields, operators):
        self.fields = fields
        self.ops = operators

    def predict_intermediate_cell_centered_velocity(self):
        """
        Step 1: Predict intermediate cell-centered velocities u*, v* using AB2 for convection and CN2 for diffusion.
        """
        Re = self.fields.grid.Re
        dt = self.fields.grid.dt
        dx = self.fields.grid.dx
        dy = self.fields.grid.dy

        # AB2 convective terms
        div_Uu_n   = self.ops.divergence(np.multiply(self.fields.U, self.ops.interpolate_center_to_face(self.fields.u, axis=0)), np.multiply(self.fields.V, self.ops.interpolate_center_to_face(self.fields.u, axis=1)))
        div_Uu_nm1 = self.ops.divergence(np.multiply(self.fields.U_prev, self.ops.interpolate_center_to_face(self.fields.u_prev, axis=0)), np.multiply(self.fields.V_prev, self.ops.interpolate_center_to_face(self.fields.u_prev, axis=1)))
        div_Uv_n   = self.ops.divergence(np.multiply(self.fields.U, self.ops.interpolate_center_to_face(self.fields.v, axis=0)), np.multiply(self.fields.V, self.ops.interpolate_center_to_face(self.fields.v, axis=1)))
        div_Uv_nm1 = self.ops.divergence(np.multiply(self.fields.U_prev, self.ops.interpolate_center_to_face(self.fields.v_prev, axis=0)), np.multiply(self.fields.V_prev, self.ops.interpolate_center_to_face(self.fields.v_prev, axis=1)))        

        # CN2 Laplacians
        lap_u_n = self.ops.laplacian(self.fields.u)
        lap_v_n = self.ops.laplacian(self.fields.v)

        # RHS terms
        explicit_U = (1/dt) * self.fields.u - (1.5 * div_Uu_n - 0.5 * div_Uu_nm1) + (1/(2 * Re)) * lap_u_n
        explicit_V = (1/dt) * self.fields.v - (1.5 * div_Uv_n - 0.5 * div_Uv_nm1) + (1/(2 * Re)) * lap_v_n
        explicit_term = [explicit_U, explicit_V]
        initial_guess = [self.fields.u, self.fields.v]

        # Discretization coefficients
        a_e = a_w = 1 / (2 * Re * dx**2)
        a_n = a_s = 1 / (2 * Re * dy**2)
        a_p = (1 / dt) + (a_e + a_w + a_n + a_s)

        # Solve for u*, v*
        u_star, v_star, _, _, _ = self.ops.gs_for_velocity(a_e, a_w, a_n, a_s, a_p, explicit_term, initial_guess)

        return u_star, v_star

    def interpolate_cell_centered_velocity_to_faces(self, u_star, v_star):
        """
        Step 2: Interpolate intermediate cell-centered velocities to face-centered.
        """
        U_star = self.ops.interpolate_center_to_face(u_star, axis=0)
        V_star = self.ops.interpolate_center_to_face(v_star, axis=1)
        return U_star, V_star

    def compute_pressure_rhs(self, U_star, V_star):
        """
        Step 3 (part 1): Compute RHS of pressure Poisson equation: div(U*) / dt of cell face velocities.
        """
        dt = self.fields.grid.dt
        rhs_p = (1 / dt) * self.ops.divergence(U_star, V_star)
        return rhs_p

    def solve_pressure_poisson(self, rhs_p):
        """
        Step 3 (part 2): Solve the pressure Poisson equation using the GS method, using cell face velocities.
        """
        dx = self.ops.dx
        dy = self.ops.dy

        a_e = a_w = -1.0 / dx**2
        a_n = a_s = -1.0 / dy**2
        a_p = -2.0 * (1.0 / dx**2 + 1.0 / dy**2)

        self.fields.p_prev = self.fields.p.copy()
        self.fields.p, _, _ = self.ops.gs_for_pressure(a_e, a_w, a_n, a_s, a_p, rhs_p, self.fields.p_prev) 

    def correct_cell_center_velocity(self, u_star, v_star): # TODO CHECK THESE ESPECAILLY HOW PRESSURE GRAD is computed
        """
        Step 4 (part 1): Correct cell-centered velocities using pressure gradient. Use given pressure (not interpolated).
        """
        dt = self.fields.grid.dt

        # Set previous time step values for u and v
        self.fields.u_prev = self.fields.u.copy()
        self.fields.v_prev = self.fields.v.copy()

        # Update cell-centered velocities using pressure gradient
        grad_p_x, grad_p_y = self.ops.gradient(self.fields.p, scheme='noncompact') 
        self.fields.u = u_star - dt * grad_p_x
        self.fields.v = v_star - dt * grad_p_y

    def correct_face_velocity(self, U_star, V_star):
        """
        Step 4 (part 2): Correct face-centered velocities using pressure gradient (pressure interpolated to faces).
        """
        dt = self.fields.grid.dt

        # Set previous time step values for U and V
        self.fields.U_prev = self.fields.U.copy()
        self.fields.V_prev = self.fields.V.copy()

        # Update face-centered velocities using pressure gradient
        grad_p_x, grad_p_y = self.ops.gradient(self.fields.p, scheme='compact')

        # Correct face-centered velocities
        self.fields.U = U_star - dt * grad_p_x
        self.fields.V = V_star - dt * grad_p_y

    def check_divergence_free(self, u, v):
        """
        Check if the final velocity field is divergence-free (∇·u ≈ 0).
        Returns max absolute divergence.
        """
        div_uv = self.ops.divergence(u, v)
        return np.max(np.abs(div_uv))

    def advance_one_step(self):
        """
        Perform one full fractional-step integration step.
        Returns: max divergence (to check incompressibility).
        """
        u_star, v_star = self.predict_intermediate_cell_centered_velocity()
        U_star, V_star = self.interpolate_cell_centered_velocity_to_faces(u_star, v_star)
        rhs_p = self.compute_pressure_rhs(U_star, V_star)
        #div_before_ppe = self.check_divergence_free(self.fields.U, self.fields.V)
        self.solve_pressure_poisson(rhs_p)
        self.correct_cell_center_velocity(u_star, v_star)
        self.correct_face_velocity(U_star, V_star)
        div_after_ppe = self.check_divergence_free(self.fields.U, self.fields.V)
        print(str(div_after_ppe) + ' after PPE')
        return self.check_divergence_free(self.fields.u, self.fields.v)
    

