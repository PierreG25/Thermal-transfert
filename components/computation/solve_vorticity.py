import numpy as np
from components.computation.thomas_algorithm import solve_thomas

def solve_adi_w(w, T, psi, u, v, diff_coeff, source_coeff, dt, dx, dy, U0):
    Ny, Nx = w.shape
    w_new = w.copy()
    s_half = (dt / 2.0) * source_coeff
    
    # --- Étape 1 : X-Implicite ---
    w_star = w.copy()
    w_star[:, 0] = -2 * psi[:, 1] / dx**2
    w_star[:, -1] = -2 * psi[:, -2] / dx**2
    
    # Coeffs
    Fx = (diff_coeff * dt) / (2 * dx**2)
    Fy = (diff_coeff * dt) / (2 * dy**2)

    for j in range(1, Ny-1):

        Cx = u[j, :] * dt / (4 * dx)
        Cy = v[j, :] * dt / (4 * dy)

        a = np.full(Nx-3, -Fx - Cx[2:-1])
        b = np.full(Nx-2, 1 + 2*Fx)
        c = np.full(Nx-3, -Fx + Cx[1:-2])

        d = w[j, 1:-1] + Fy*(w[j+1, 1:-1] - 2*w[j, 1:-1] + w[j-1, 1:-1]) \
            - (Cy[1:-1])*(w[j+1, 1:-1] - w[j-1, 1:-1])

        d += s_half * (T[j, 2:]-T[j, :-2]) / (2*dx)  # dT/dx

        d[0] -= a[0]*w_star[j, 0]
        d[-1] -= c[-1]*w_star[j, -1]
        w_star[j, 1:-1] = solve_thomas(a, b, c, d)

    # --- Étape 2 : Y-Implicite ---
    w_star[0, :] = -2 * psi[1, :] / dy**2
    w_star[-1, :] = -2 * (psi[-2, :] + U0 * dy) / dy**2
    

    for i in range(1, Nx-1):

        Cx = u[:, i] * dt / (4 * dx)
        Cy = v[:, i] * dt / (4 * dy)

        a = np.full(Ny-3, -Fy - Cy[1:-2])
        b = np.full(Ny-2, 1 + 2*Fy)
        c = np.full(Ny-3, -Fy + Cy[2:-1])
        d = w_star[1:-1, i] + Fx*(w_star[1:-1, i+1] - 2*w_star[1:-1, i] + w_star[1:-1, i-1]) \
            - (Cx[1:-1])*(w_star[1:-1, i+1] - w_star[1:-1, i-1])

        d += s_half * (T[1:-1, i+1] - T[1:-1, i-1]) / (2*dx)  # dT/dx

        d[0] -= a[0]*w_star[0, i]
        d[-1] -= c[-1]*w_star[-1, i]
        w_new[1:-1, i] = solve_thomas(a, b, c, d)

    return w_new