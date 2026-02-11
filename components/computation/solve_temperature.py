import numpy as np
from components.computation.thomas_algorithm import solve_thomas_vectorized

def solve_adi_T(T, u, v, diff_coeff, dt, dx, dy, work_buffer=None):
    
    # --- 1. Initialisation ---
    if work_buffer is not None:
        T_new = work_buffer['T_new']
        T_star = work_buffer['T_star']
    else:
        T_new = np.zeros_like(T)
        T_star = np.zeros_like(T)

    # Paramètres physiques
    Fx = (diff_coeff * dt) / (2 * dx**2)
    Fy = (diff_coeff * dt) / (2 * dy**2)
    
    # =================================================================
    # ÉTAPE 0 : BCs DIRICHLET (Gauche/Droite)
    # =================================================================
    T_star[:, 0]  = T[:, 0]
    T_star[:, -1] = T[:, -1]
    T_new[:, 0]   = T[:, 0]
    T_new[:, -1]  = T[:, -1]

    # ==========================================
    # ÉTAPE 1 : X-IMPLICITE (Résolution lignes horizontales)
    # ==========================================
    
    # --- Matrices LHS (A, B, C) ---
    u_x = u[:, 1:-1] # (Ny, Nx-2)
    
    Pex = np.abs(u_x) * dx / diff_coeff
    mask = Pex >= 2
    
    Cx = u_x * dt / (4 * dx)
    
    # Coefficients Hybrides
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
    d = T[:, 1:-1].copy() # (Ny, Nx-2)
    
    # Diffusion Y
    diff_y = Fy * (T[2:, 1:-1] - 2*T[1:-1, 1:-1] + T[:-2, 1:-1])
    
    # Convection Y
    v_y = v[1:-1, 1:-1]
    Pey = np.abs(v_y) * dy / diff_coeff
    mask_y = Pey >= 2
    
    Cy = v_y * dt / (4 * dy)
    conv_y_c = Cy * (T[2:, 1:-1] - T[:-2, 1:-1])
    
    vp, vm = np.maximum(v_y, 0), np.minimum(v_y, 0)
    Cuy = dt / (2 * dy)
    conv_y_u = Cuy * (vp*(T[1:-1, 1:-1] - T[:-2, 1:-1]) + vm*(T[2:, 1:-1] - T[1:-1, 1:-1]))
    
    conv_y = np.where(mask_y, conv_y_u, conv_y_c)
    
    # Update intérieur de d
    d[1:-1, :] = d[1:-1, :] + diff_y - conv_y
    
    # Injection CL Neumann
    d[0, :]  = T[0, 1:-1]  + Fy*(2*T[1, 1:-1] - 2*T[0, 1:-1])   # Bas
    d[-1, :] = T[-1, 1:-1] + Fy*(2*T[-2, 1:-1] - 2*T[-1, 1:-1]) # Haut
    
    # Injection CL Dirichlet
    d[:, 0]  -= a[:, 0]  * T_star[:, 0]
    d[:, -1] -= c[:, -1] * T_star[:, -1]
    
    # Résolution
    T_star[:, 1:-1] = solve_thomas_vectorized(a, b, c, d)
    
    
    # ==========================================
    # ÉTAPE 2 : Y-IMPLICITE (Résolution colonnes verticales)
    # ==========================================
    
    T_trans = T_star.T
    u_trans = u.T
    v_trans = v.T
    
    # --- Matrices LHS ---
    v_y = v_trans[1:-1, :]
    
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
    b = np.where(mask_y, b_u, b_c) 
    c = np.where(mask_y, c_u, c_c)
    
    # --- Condition flux nul par maille fictive ---
    c[:, 0]  += a[:, 0]
    a[:, -1] += c[:, -1]
    
    # --- RHS ---
    d_final = T_trans[1:-1, :].copy()
    
    # Diffusion X
    diff_x = Fx * (T_trans[2:, :] - 2*T_trans[1:-1, :] + T_trans[:-2, :])
    
    # Convection X
    u_x = u_trans[1:-1, :]
    Pex = np.abs(u_x) * dx / diff_coeff
    mask_x = Pex >= 2
    
    Cx = u_x * dt / (4 * dx)
    conv_x_c = Cx * (T_trans[2:, :] - T_trans[:-2, :])
    
    up, um = np.maximum(u_x, 0), np.minimum(u_x, 0)
    Cux = dt / (2 * dx)
    conv_x_u = Cux * (up*(T_trans[1:-1, :] - T_trans[:-2, :]) + um*(T_trans[2:, :] - T_trans[1:-1, :]))
    
    conv_x = np.where(mask_x, conv_x_u, conv_x_c)
    
    # Mise à jour du RHS
    d_final[:, :] = d_final[:, :] + diff_x - conv_x
    
    # Résolution
    res_y = solve_thomas_vectorized(a, b, c, d_final)
    
    # Reconstitution T_new
    T_new_trans = T_trans.copy()
    T_new_trans[1:-1, :] = res_y 
    
    T_new[:, :] = T_new_trans.T
    
    return T_new