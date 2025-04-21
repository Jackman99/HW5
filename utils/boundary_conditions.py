def apply_velocity_bc(u, v, lid_velocity=1.0):
    u[:, 0] = -u[:, 1]
    u[:, -1] = 2 * lid_velocity - u[:, -2]
    u[0, :] = -u[1, :]
    u[-1, :] = -u[-2, :]

    v[:, 0] = -v[:, 1]
    v[:, -1] = -v[:, -2]
    v[0, :] = -v[1, :]
    v[-1, :] = -v[-2, :]

def apply_pressure_bc(p):
    p[0, :] = p[1, :]
    p[-1, :] = p[-2, :]
    p[:, 0] = p[:, 1]
    p[:, -1] = p[:, -2]