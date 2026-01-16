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
    gamma_x = (diff_coeff * dt) / (2 * dx**2)
    gamma_y = (diff_coeff * dt) / (2 * dy**2)
    beta_x = dt / (4 * dx)
    beta_y = dt / (4 * dy)

    for j in range(1, Ny-1):
        a = np.full(Nx-3, -gamma_x - u[j, 1:-2]*beta_x)
        b = np.full(Nx-2, 1 + 2*gamma_x)
        c = np.full(Nx-3, -gamma_x + u[j, 2:-1]*beta_x)
        
        # Partie explicite en Y
        d = T[j, 1:-1] + gamma_y*(T[j+1, 1:-1] - 2*T[j, 1:-1] + T[j-1, 1:-1]) \
            - (v[j, 1:-1]*beta_y)*(T[j+1, 1:-1] - T[j-1, 1:-1])

        d[0] -= a[0]*T_star[j, 0]
        d[-1] -= c[-1]*T_star[j, -1]
        T_star[j, 1:-1] = solve_thomas(a, b, c, d)

    # --- Étape 2 : Y-Implicite ---
    # CL Neumann (Haut/Bas)
    T_star[0, :] = (4*T_star[1, :] - T_star[2, :]) / 3
    T_star[-1, :] = (4*T_star[-2, :] - T_star[-3, :]) / 3
    

    for i in range(1, Nx-1):
        a = np.full(Ny-3, -gamma_y - v[1:-2, i]*beta_y)
        b = np.full(Ny-2, 1 + 2*gamma_y)
        c = np.full(Ny-3, -gamma_y + v[2:-1, i]*beta_y)

        d = T_star[1:-1, i] + gamma_x*(T_star[1:-1, i+1] - 2*T_star[1:-1, i] + T_star[1:-1, i-1]) \
            - (u[1:-1, i]*beta_x)*(T_star[1:-1, i+1] - T_star[1:-1, i-1])

        d[0] -= a[0]*T_new[0, i]
        d[-1] -= c[-1]*T_new[-1, i]
        T_new[1:-1, i] = solve_thomas(a, b, c, d)

    return T_new