import numpy as np
from components.computation.thomas_algorithm import solve_thomas

def solve_adi_T(T, u, v, diff_coeff, dt, dx, dy):
    Ny, Nx = T.shape
    T_new = T.copy()
    
    # --- Étape 1 : X-Implicite ---
    T_star = T.copy()
    # CL Dirichlet (Côtés)
    T_star[:, 0] = 1.0
    T_star[:, -1] = 0.0

    # Coeffs

    Fx = (diff_coeff * dt) / (2 * dx**2)
    Fy = (diff_coeff * dt) / (2 * dy**2)

    for j in range(Ny):

        Cx = u[j, :] * dt / (4 * dx)
        Cy = v[j, :] * dt / (4 * dy)

        a = -Fx - Cx[2:-1].copy()
        b = np.full(Nx-2, 1 + 2*Fx)
        c = -Fx + Cx[1:-2].copy()

        # Partie explicite en Y
        if j == 0:
            d = T[j, 1:-1] + Fy*(2*T[j+1, 1:-1] - 2*T[j, 1:-1])
        elif j == Ny-1:
            d = T[j, 1:-1] + Fy*(2*T[j-1, 1:-1] - 2*T[j, 1:-1])
        else:
            d = T[j, 1:-1] + Fy*(T[j+1, 1:-1] - 2*T[j, 1:-1] + T[j-1, 1:-1]) \
                - (Cy[1:-1])*(T[j+1, 1:-1] - T[j-1, 1:-1])

        a_ext = -Fx - Cx[1]
        d[0] -= a_ext * T_star[j, 0]

        # Bord droit (i=Nx-1) : T_star[j, -1] = 0.0
        c_ext = -Fx + Cx[-2]
        d[-1] -= c_ext * T_star[j, -1]

        T_star[j, 1:-1] = solve_thomas(a, b, c, d)

    # --- Étape 2 : Y-Implicite ---
    # CL Neumann (Haut/Bas)

    for i in range(1, Nx-1):

        Cx = u[:, i] * dt / (4 * dx)
        Cy = v[:, i] * dt / (4 * dy)

        a = -Fy - Cy[:-2].copy()
        a = np.append(a, -2*Fy)  # Neumann en bas

        b = np.full(Ny, 1 + 2*Fy)

        c = -Fy + Cy[1:-1].copy()
        c = np.insert(c, 0, -2*Fy)  # Neumann en haut   

        d = T_star[:, i] + Fx*(T_star[:, i+1] - 2*T_star[:, i] + T_star[:, i-1]) \
            - (Cx[:])*(T_star[:, i+1] - T_star[:, i-1])

        T_new[:, i] = solve_thomas(a, b, c, d)

    return T_new