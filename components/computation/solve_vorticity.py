import numpy as np
from components.computation.thomas_algorithm import solve_thomas

def solve_adi_w(w, T, psi, u, v, diff_coeff, source_coeff, dt, dx, dy, U0):
    Ny, Nx = w.shape
    w_new = w.copy()
    w_star = w.copy()
    s_half = (dt / 2.0) * source_coeff
    
    # Coeffs de diffusion
    Fx = (diff_coeff * dt) / (2 * dx**2)
    Fy = (diff_coeff * dt) / (2 * dy**2)
    
    # --- Étape 1 : X-Implicite ---
    # Conditions aux limites (Thom) pour les parois verticales
    w_star[:, 0] = -2 * psi[:, 1] / dx**2
    w_star[:, -1] = -2 * psi[:, -2] / dx**2
    
    for j in range(1, Ny-1):
        uj = u[j, 1:-1]
        Pex_local = (np.abs(uj) * dx) / diff_coeff
        mask_up_x = Pex_local >= 2
        
        # 1. Matrice X (Implicite)
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

        # 2. Second membre d (Explicite Y Hybride + Source Boussinesq)
        vj = v[j, 1:-1]
        Pey_local = (np.abs(vj) * dy) / diff_coeff
        
        # Convection Y
        Cy_cent = vj * dt / (4 * dy)
        conv_y_cent = Cy_cent * (w[j+1, 1:-1] - w[j-1, 1:-1])
        vp_y, vm_y = np.maximum(vj, 0), np.minimum(vj, 0)
        Cuy_up = dt / (2 * dy)
        conv_y_up = Cuy_up * (vp_y * (w[j, 1:-1] - w[j-1, 1:-1]) + vm_y * (w[j+1, 1:-1] - w[j, 1:-1]))
        
        conv_y = np.where(Pey_local >= 2, conv_y_up, conv_y_cent)
        
        d = w[j, 1:-1] + Fy * (w[j+1, 1:-1] - 2*w[j, 1:-1] + w[j-1, 1:-1]) - conv_y
        
        # Terme source (Boussinesq) : dT/dx
        d += s_half * (T[j, 2:] - T[j, :-2]) / (2 * dx)

        # 3. Injection Dirichlet (Thom sur parois verticales)
        if Pex_local[0] >= 2:
            d[0] -= a_up[0] * w_star[j, 0]
        else:
            d[0] -= a_cent[0] * w_star[j, 0]

        if Pex_local[-1] >= 2:
            d[-1] -= c_up[-1] * w_star[j, -1]
        else:
            d[-1] -= c_cent[-1] * w_star[j, -1]
        
        w_star[j, 1:-1] = solve_thomas(a, b, c, d)

    # --- Étape 2 : Y-Implicite ---
    # Conditions aux limites (Thom) pour parois horizontales (avec U0 en haut)
    w_star[0, :] = -2 * psi[1, :] / dy**2
    w_star[-1, :] = -2 * (psi[-2, :] + U0 * dy) / dy**2
    
    for i in range(1, Nx-1):
        vi = v[1:-1, i]
        Pey_local = (np.abs(vi) * dy) / diff_coeff
        mask_up_y = Pey_local >= 2
        
        # 1. Matrice Y (Implicite)
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

        # 2. Second membre d (Explicite X Hybride + Source)
        ui = u[1:-1, i]
        Pex_local_i = (np.abs(ui) * dx) / diff_coeff
        
        # Convection X
        Cx_cent = ui * dt / (4 * dx)
        conv_x_cent = Cx_cent * (w_star[1:-1, i+1] - w_star[1:-1, i-1])
        up_x, um_x = np.maximum(ui, 0), np.minimum(ui, 0)
        Cux_up = dt / (2 * dx)
        conv_x_up = Cux_up * (up_x * (w_star[1:-1, i] - w_star[1:-1, i-1]) + um_x * (w_star[1:-1, i+1] - w_star[1:-1, i]))
        
        conv_x = np.where(Pex_local_i >= 2, conv_x_up, conv_x_cent)
        
        d_y = w_star[1:-1, i] + Fx * (w_star[1:-1, i+1] - 2*w_star[1:-1, i] + w_star[1:-1, i-1]) - conv_x
        
        # Ajout du terme source Boussinesq (dT/dx)
        d_y += s_half * (T[1:-1, i+1] - T[1:-1, i-1]) / (2 * dx)

        # 3. Injection Dirichlet (Thom sur parois horizontales)
        if Pey_local[0] >= 2:
            d_y[0] -= a_up_y[0] * w_star[0, i]
        else:
            d_y[0] -= a_cent_y[0] * w_star[0, i]

        if Pey_local[-1] >= 2:
            d_y[-1] -= c_up_y[-1] * w_star[-1, i]
        else:
            d_y[-1] -= c_cent_y[-1] * w_star[-1, i]

        w_new[1:-1, i] = solve_thomas(a_y, b_y, c_y, d_y)

    return w_new