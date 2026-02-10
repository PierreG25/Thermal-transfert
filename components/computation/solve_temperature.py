import numpy as np
from components.computation.thomas_algorithm import solve_thomas_vectorized

def solve_adi_T(T, u, v, diff_coeff, dt, dx, dy, work_buffer=None):
    """Version vectorisée optimisée de ADI Température"""

# --- GESTION MÉMOIRE ---
    if work_buffer is not None:
        T_new = work_buffer['T_new']
        T_star = work_buffer['T_star']
    else:
        T_new = np.zeros_like(T)
        T_star = np.zeros_like(T)

    T_star[:] = T[:] 
    
    # Coefficients constants
    Fx = (diff_coeff * dt) / (2 * dx**2)
    Fy = (diff_coeff * dt) / (2 * dy**2)    

    # ===================================================
    # ÉTAPE 1 : X-Implicite
    # ===================================================
    # On travaille sur l'intérieur [1:-1]
    
    u_x = u[:, 1:-1] 
    
    # Nombre de Peclet local (Ny, Nx-2)
    Pex = np.abs(u_x) * dx / diff_coeff
    mask = Pex >= 2
    
    # Coefficients Schéma Centré (matrices pleines (Ny, Nx-2))
    Cx = u_x * dt / (4 * dx)
    a_c = -Fx - Cx
    b_c = 1 + 2*Fx
    c_c = -Fx + Cx
    
    # Coefficients Schéma Upwind
    up, um = np.maximum(u_x, 0), np.minimum(u_x, 0)
    Cux = dt / (2 * dx)
    a_u = -Fx - up * Cux
    b_u = 1 + 2*Fx + (up - um) * Cux
    c_u = -Fx + um * Cux
    
    # Sélection (Hybride)
    a = np.where(mask, a_u, a_c)
    b = np.where(mask, b_u, b_c)
    c = np.where(mask, c_u, c_c)
    
    # 2. Second membre d (Explicite en Y)
    d = T[:, 1:-1].copy()
    
    diff_y = Fy * (T[2:, 1:-1] - 2*T[1:-1, 1:-1] + T[:-2, 1:-1])
    
    # Convection Y
    v_y = v[1:-1, 1:-1] # Vitesse verticale intérieure
    Pey = np.abs(v_y) * dy / diff_coeff
    mask_y = Pey >= 2
    
    # Centré Y
    Cy = v_y * dt / (4 * dy)
    conv_y_c = Cy * (T[2:, 1:-1] - T[:-2, 1:-1])
    
    # Upwind Y
    vp, vm = np.maximum(v_y, 0), np.minimum(v_y, 0)
    Cuy = dt / (2 * dy)
    conv_y_u = Cuy * (vp * (T[1:-1, 1:-1] - T[:-2, 1:-1]) + vm * (T[2:, 1:-1] - T[1:-1, 1:-1]))
    
    conv_y = np.where(mask_y, conv_y_u, conv_y_c)
    
    # Application sur d
    d[1:-1, :] = d[1:-1, :] + diff_y - conv_y
    
    # Maille fictive
    d[0, :]  = T[0, 1:-1]  + Fy*(2*T[1, 1:-1] - 2*T[0, 1:-1])
    d[-1, :] = T[-1, 1:-1] + Fy*(2*T[-2, 1:-1] - 2*T[-1, 1:-1])
    
    # 3. Injection Conditions Limites X (Dirichlet T_star)
    d[:, 0]  -= a[:, 0] * T[:, 0] 
    d[:, -1] -= c[:, -1] * T[:, -1]
    
    # 4. Résolution Thomas 
    res = solve_thomas_vectorized(a, b, c, d)
    
    T_star[:, 0] = T[:, 0]    # BC Gauche
    T_star[:, -1] = T[:, -1]  # BC Droite
    T_star[:, 1:-1] = res     # Intérieur calculé
    
    
    # ===================================================
    # ÉTAPE 2 : Y-Implicite (Résolution par colonnes)
    # ===================================================
    
    T_trans = T_star.T       # (Nx, Ny)
    u_trans = u.T            # (Nx, Ny)
    v_trans = v.T            # (Nx, Ny)
    
    # On travaille sur l'intérieur Y (donc indices 1:-1 dans la version transposée)
    v_y = v_trans[:, 1:-1]
    
    # Peclet Y
    Pey = np.abs(v_y) * dy / diff_coeff
    mask_y = Pey >= 2
    
    # Coefficients Y (Implicites)
    Cy = v_y * dt / (4 * dy)
    a_c = -Fy - Cy
    b_c = 1 + 2*Fy
    c_c = -Fy + Cy
    
    vp, vm = np.maximum(v_y, 0), np.minimum(v_y, 0)
    Cuy = dt / (2 * dy)
    a_u = -Fy - vp * Cuy
    b_u = 1 + 2*Fy + (vp - vm) * Cuy
    c_u = -Fy + vm * Cuy
    
    a = np.where(mask_y, a_u, a_c)
    b = np.where(mask_y, b_u, b_c)
    c = np.where(mask_y, c_u, c_c)
    
    # Maille fictive
    c[:, 0]  += a[:, 0]
    a[:, -1] += c[:, -1]
    
    d_y = T_trans[:, 1:-1].copy()


    diff_x = Fx * (T_trans[2:, 1:-1] - 2*T_trans[1:-1, 1:-1] + T_trans[:-2, 1:-1])
    
    u_x = u_trans[1:-1, 1:-1]
    Pex = np.abs(u_x) * dx / diff_coeff
    mask_x = Pex >= 2
    
    Cx = u_x * dt / (4 * dx)
    conv_x_c = Cx * (T_trans[2:, 1:-1] - T_trans[:-2, 1:-1])
    
    up, um = np.maximum(u_x, 0), np.minimum(u_x, 0)
    Cux = dt / (2 * dx)
    conv_x_u = Cux * (up*(T_trans[1:-1, 1:-1] - T_trans[:-2, 1:-1]) + um*(T_trans[2:, 1:-1] - T_trans[1:-1, 1:-1]))
    
    conv_x = np.where(mask_x, conv_x_u, conv_x_c)
    d_y[1:-1, :] = d_y[1:-1, :] + diff_x - conv_x
    
    
    # Restriction aux i intérieurs pour la résolution
    a_s = a[1:-1, :]
    b_s = b[1:-1, :]
    c_s = c[1:-1, :]
    d_s = d_y[1:-1, :]
    
    # Résolution vectorisée
    res_y = solve_thomas_vectorized(a_s, b_s, c_s, d_s) # (Nx-2, Ny-2)
    
    # Reconstitution T_new (Transposé)
    T_new_trans = T_trans.copy()
    T_new_trans[1:-1, 1:-1] = res_y
    
    # Transposition inverse
    T_new = T_new_trans.T
    
    return T_new