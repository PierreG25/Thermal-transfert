import numpy as np
from components.computation.thomas_algorithm import solve_thomas_vectorized

def solve_adi_w(w, T, psi, u, v, diff_coeff, Ri, dt, dx, dy, U0, work_buffer=None):
    
    # --- 1. Initialisation des buffers ---
    if work_buffer is not None:
        w_new = work_buffer['w_new']
        w_star = work_buffer['w_star']
    else:
        w_new = np.zeros_like(w)
        w_star = np.zeros_like(w)

    # Paramètres physiques
    Fx = (diff_coeff * dt) / (2 * dx**2)
    Fy = (diff_coeff * dt) / (2 * dy**2)
    s_half = (dt / 2.0) * Ri 

    # =================================================================
    # ÉTAPE 0 : CONDITIONS LIMITES DIRICHLET
    # =================================================================
    
    w_wall_left  = -2 * psi[:, 1]  / dx**2
    w_wall_right = -2 * psi[:, -2] / dx**2
    w_wall_bottom = -2 * psi[1, :] / dy**2
    w_wall_top    = -2 * (psi[-2, :] + U0 * dy) / dy**2

    w_star[:, 0]  = w_wall_left
    w_star[:, -1] = w_wall_right
    w_star[0, :]  = w_wall_bottom
    w_star[-1, :] = w_wall_top

    w_new[:, 0]  = w_wall_left
    w_new[:, -1] = w_wall_right
    w_new[0, :]  = w_wall_bottom
    w_new[-1, :] = w_wall_top

    # ==========================================
    # ÉTAPE 1 : X-IMPLICITE (Résolution de w_star intérieur)
    # ==========================================
    
    # --- Matrices LHS (A, B, C) ---
    u_x = u[1:-1, 1:-1] 
    Pex = np.abs(u_x) * dx / diff_coeff
    mask = Pex >= 2
    
    Cx = u_x * dt / (4 * dx)
    
    # Schémas centré / Upwind (Hybride)
    a_c, b_c, c_c = -Fx - Cx, 1 + 2*Fx, -Fx + Cx
    
    up, um = np.maximum(u_x, 0), np.minimum(u_x, 0)
    Cux = dt / (2 * dx)
    a_u = -Fx - up * Cux
    b_u = 1 + 2*Fx + (up - um) * Cux
    c_u = -Fx + um * Cux
    
    a = np.where(mask, a_u, a_c)
    b = np.where(mask, b_u, b_c)
    c = np.where(mask, c_u, c_c)
    
    # --- Second membre RHS (d) ---
    w_inner = w[1:-1, 1:-1]
    d = w_inner.copy()
    
    # Diffusion Y
    diff_y = Fy * (w[2:, 1:-1] - 2*w_inner + w[:-2, 1:-1])
    
    # Convection Y (Centré / Upwind)
    v_y = v[1:-1, 1:-1]
    Pey = np.abs(v_y) * dy / diff_coeff
    mask_y = Pey >= 2
    
    Cy = v_y * dt / (4 * dy)
    conv_y_c = Cy * (w[2:, 1:-1] - w[:-2, 1:-1])
    
    vp, vm = np.maximum(v_y, 0), np.minimum(v_y, 0)
    Cuy = dt / (2 * dy)
    conv_y_u = Cuy * (vp*(w_inner - w[:-2, 1:-1]) + vm*(w[2:, 1:-1] - w_inner))
    
    conv_y = np.where(mask_y, conv_y_u, conv_y_c)
    
    # Terme source
    dTdx = (T[1:-1, 2:] - T[1:-1, :-2]) / (2*dx)
    source = s_half * dTdx
    
    d = d + diff_y - conv_y + source
    
    # Injection des CL Dirichlet
    d[:, 0]  -= a[:, 0]  * w_star[1:-1, 0]
    d[:, -1] -= c[:, -1] * w_star[1:-1, -1]
    
    # --- Résolution ---
    w_star[1:-1, 1:-1] = solve_thomas_vectorized(a, b, c, d)
    
    
    # ==========================================
    # ÉTAPE 2 : Y-IMPLICITE (Résolution de w_new intérieur)
    # ==========================================
    
    w_trans = w_star.T 
    w_new_trans = w_new.T 
    
    u_trans, v_trans, T_trans = u.T, v.T, T.T
    
    # --- Matrices LHS ---
    v_y = v_trans[1:-1, 1:-1]
    Pey = np.abs(v_y) * dy / diff_coeff
    mask_y = Pey >= 2
    
    Cy = v_y * dt / (4 * dy)
    a_c, b_c, c_c = -Fy - Cy, 1 + 2*Fy, -Fy + Cy
    
    vp, vm = np.maximum(v_y, 0), np.minimum(v_y, 0)
    Cuy = dt / (2 * dy)
    a_u = -Fy - vp * Cuy
    b_u = 1 + 2*Fy + (vp - vm) * Cuy
    c_u = -Fy + vm * Cuy
    
    a = np.where(mask_y, a_u, a_c)
    b = np.where(mask, b_u, b_c)
    b = np.where(mask_y, b_u, b_c)
    c = np.where(mask_y, c_u, c_c)
    
    # --- RHS ---
    w_star_inner_T = w_trans[1:-1, 1:-1]
    d_y = w_star_inner_T.copy()
    
    # Diffusion X
    diff_x = Fx * (w_trans[2:, 1:-1] - 2*w_star_inner_T + w_trans[:-2, 1:-1])
    
    # Convection X
    u_x = u_trans[1:-1, 1:-1]
    Pex = np.abs(u_x) * dx / diff_coeff
    mask_x = Pex >= 2
    
    Cx = u_x * dt / (4 * dx)
    conv_x_c = Cx * (w_trans[2:, 1:-1] - w_trans[:-2, 1:-1])
    
    up, um = np.maximum(u_x, 0), np.minimum(u_x, 0)
    Cux = dt / (2 * dx)
    conv_x_u = Cux * (up*(w_star_inner_T - w_trans[:-2, 1:-1]) + um*(w_trans[2:, 1:-1] - w_star_inner_T))
    
    conv_x = np.where(mask_x, conv_x_u, conv_x_c)
    
    # Source
    dTdx_trans = (T_trans[2:, 1:-1] - T_trans[:-2, 1:-1]) / (2*dx)
    source = s_half * dTdx_trans
    
    d_y = d_y + diff_x - conv_x + source
    
    # Injection des CL Dirichlet
    # w_new_trans[:, 0] correspond à la paroi du bas (y=0) qui est maintenant la gauche de la matrice transposée
    d_y[:, 0]  -= a[:, 0]  * w_new_trans[1:-1, 0]
    d_y[:, -1] -= c[:, -1] * w_new_trans[1:-1, -1]
    
    # --- Résolution ---
    w_new_trans[1:-1, 1:-1] = solve_thomas_vectorized(a, b, c, d_y)
    
    w_new[:, :] = w_new_trans.T
    
    return w_new