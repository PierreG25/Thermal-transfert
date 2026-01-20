import numpy as np
from components.computation.thomas_algorithm import solve_thomas

def solve_adi_T(T, u, v, diff_coeff, dt, dx, dy):
    Ny, Nx = T.shape
    T_new = T.copy()
    T_star = T.copy()
    
    # Coefficients de diffusion (pour un demi-pas dt/2)
    Fx = (diff_coeff * dt) / (2 * dx**2)
    Fy = (diff_coeff * dt) / (2 * dy**2)
    
    # --- Étape 1 : X-Implicite ---
    for j in range(Ny):
        uj = u[j, 1:-1]
        Pex_local = (np.abs(uj) * dx) / diff_coeff
        mask_up_x = Pex_local >= 2
        
        # 1. Matrices X (Implicite)
        Cx_cent = uj * dt / (4 * dx)
        a_cent = -Fx - Cx_cent
        b_cent = 1 + 2*Fx
        c_cent = -Fx + Cx_cent
        
        up_x, um_x = np.maximum(uj, 0), np.minimum(uj, 0)
        Cux_up = dt / (2 * dx)
        a_up = -Fx - up_x * Cux_up
        b_up = 1 + 2*Fx + (up_x - um_x) * Cux_up
        c_up = -Fx + um_x * Cux_up
        
        a = np.where(mask_up_x[1:], a_up[1:], a_cent[1:])
        b = np.where(mask_up_x, b_up, b_cent)
        c = np.where(mask_up_x[:-1], c_up[:-1], c_cent[:-1])

        # 2. Second membre d (Explicite Y Hybride)
        if j == 0:
            d = T[j, 1:-1] + Fy * (2*T[j+1, 1:-1] - 2*T[j, 1:-1])
        elif j == Ny-1:
            d = T[j, 1:-1] + Fy * (2*T[j-1, 1:-1] - 2*T[j, 1:-1])
        else:
            vj = v[j, 1:-1]
            Pey_local = (np.abs(vj) * dy) / diff_coeff
            # Convection Y Centrée
            Cy_cent = vj * dt / (4 * dy)
            conv_y_cent = Cy_cent * (T[j+1, 1:-1] - T[j-1, 1:-1])
            # Convection Y Upwind
            vp_y, vm_y = np.maximum(vj, 0), np.minimum(vj, 0)
            Cuy_up = dt / (2 * dy)
            conv_y_up = Cuy_up * (vp_y * (T[j, 1:-1] - T[j-1, 1:-1]) + vm_y * (T[j+1, 1:-1] - T[j, 1:-1]))
            
            conv_y = np.where(Pey_local >= 2, conv_y_up, conv_y_cent)
            d = T[j, 1:-1] + Fy * (T[j+1, 1:-1] - 2*T[j, 1:-1] + T[j-1, 1:-1]) - conv_y

        # 3. Injection Dirichlet (Bords X)
        if Pex_local[0] >= 2:
            d[0] -= a_up[0] * T_star[j, 0]
        else:
            d[0] -= a_cent[0] * T_star[j, 0]

        if Pex_local[-1] >= 2:
            d[-1] -= c_up[-1] * T_star[j, -1]
        else:
            d[-1] -= c_cent[-1] * T_star[j, -1]
        
        T_star[j, 1:-1] = solve_thomas(a, b, c, d)

    # --- Étape 2 : Y-Implicite ---
    for i in range(1, Nx-1):
        vi = v[:, i]
        Pey_local = (np.abs(vi) * dy) / diff_coeff
        mask_up_y = Pey_local >= 2
        
        # 1. Matrices Y (Implicite)
        Cy_cent = vi * dt / (4 * dy)
        a_cent_y = -Fy - Cy_cent
        b_cent_y = 1 + 2*Fy
        c_cent_y = -Fy + Cy_cent
        
        vp_y, vm_y = np.maximum(vi, 0), np.minimum(vi, 0)
        Cuy_up = dt / (2 * dy)
        a_up_y = -Fy - vp_y * Cuy_up
        b_up_y = 1 + 2*Fy + (vp_y - vm_y) * Cuy_up
        c_up_y = -Fy + vm_y * Cuy_up
        
        a_y = np.where(mask_up_y[1:], a_up_y[1:], a_cent_y[1:])
        b_y = np.where(mask_up_y, b_up_y, b_cent_y)
        c_y = np.where(mask_up_y[:-1], c_up_y[:-1], c_cent_y[:-1])

        # 2. Gestion Neumann (Mailles fictives)
        coeff_fictif_bas = a_up_y[0] if mask_up_y[0] else a_cent_y[0]
        c_y[0] += coeff_fictif_bas
        coeff_fictif_haut = c_up_y[-1] if mask_up_y[-1] else c_cent_y[-1]
        a_y[-1] += coeff_fictif_haut

        # 3. Second membre d (Explicite X Hybride)
        uj_i = u[:, i]
        Pex_local_i = (np.abs(uj_i) * dx) / diff_coeff
        # Convection X Centrée
        Cx_cent = uj_i * dt / (4 * dx)
        conv_x_cent = Cx_cent * (T_star[:, i+1] - T_star[:, i-1])
        # Convection X Upwind
        up_x, um_x = np.maximum(uj_i, 0), np.minimum(uj_i, 0)
        Cux_up = dt / (2 * dx)
        conv_x_up = Cux_up * (up_x * (T_star[:, i] - T_star[:, i-1]) + um_x * (T_star[:, i+1] - T_star[:, i]))
        
        conv_x = np.where(Pex_local_i >= 2, conv_x_up, conv_x_cent)
        d_y = T_star[:, i] + Fx * (T_star[:, i+1] - 2*T_star[:, i] + T_star[:, i-1]) - conv_x

        T_new[:, i] = solve_thomas(a_y, b_y, c_y, d_y)

    return T_new