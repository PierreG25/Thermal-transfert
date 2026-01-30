import numpy as np
from components.computation.thomas_algorithm import solve_thomas_vectorized

def solve_adi_w(w, T, psi, u, v, diff_coeff, Ri, dt, dx, dy, U0, work_buffer=None):
    """
    Résolution ADI de la Vorticité (w).
    Structure symétrique à la température :
    - Étape 1 : w -> w_star (Implicite X)
    - Étape 2 : w_star -> w_new (Implicite Y)
    """
    Ny, Nx = w.shape
    
    # --- 0. GESTION MÉMOIRE & CONSTANTES ---
    if work_buffer is not None:
        w_new = work_buffer['w_new']
        w_star = work_buffer['w_star']
    else:
        w_new = np.zeros_like(w)
        w_star = np.zeros_like(w)
        
    # Paramètres physiques
    Fx = (diff_coeff * dt) / (2 * dx**2)
    Fy = (diff_coeff * dt) / (2 * dy**2)
    s_half = (dt / 2.0) * Ri  # Ri = Ra/(Pr*Re^2) pour le terme source thermique
    r_wall = 0.5 # Facteur de relaxation aux parois (Vital pour stabilité)

    # --- 1. PRÉ-CALCUL DES BORDS (THOM avec RELAXATION) ---
    # On met à jour les murs dans w_star directement pour s'en servir comme Condition Limite (CL)
    # Mur Gauche/Droite
    w_star[:, 0]  = (1-r_wall)*w[:, 0]  + r_wall*(-2 * psi[:, 1] / dx**2)
    w_star[:, -1] = (1-r_wall)*w[:, -1] + r_wall*(-2 * psi[:, -2] / dx**2)
    # Mur Bas/Haut
    w_star[0, :]  = (1-r_wall)*w[0, :]  + r_wall*(-2 * psi[1, :] / dy**2)
    w_star[-1, :] = (1-r_wall)*w[-1, :] + r_wall*(-2 * (psi[-2, :] + U0 * dy) / dy**2)
    
    # Note : À ce stade, w_star contient les "vieux" w à l'intérieur, et les "nouveaux" w aux bords.
    # C'est parfait pour l'étape 1.

    # ==========================================
    # ÉTAPE 1 : X-IMPLICITE (Calcul de w_star)
    # ==========================================
    # On résout sur les lignes intérieures j=1..Ny-2
    # Le système tridiagonal est de taille Nx-2 (i=1..Nx-2)
    
    # --- A. Matrices LHS (A, B, C) selon u ---
    u_x = u[1:-1, 1:-1] 
    Pex = np.abs(u_x) * dx / diff_coeff
    mask = Pex >= 2
    
    # Coefficients Centrés
    Cx = u_x * dt / (4 * dx)
    a_c, b_c, c_c = -Fx - Cx, 1 + 2*Fx, -Fx + Cx
    
    # Coefficients Upwind (si convection forte)
    up, um = np.maximum(u_x, 0), np.minimum(u_x, 0)
    Cux = dt / (2 * dx)
    a_u = -Fx - up * Cux
    b_u = 1 + 2*Fx + (up - um) * Cux
    c_u = -Fx + um * Cux
    
    # Sélection Hybride
    a = np.where(mask, a_u, a_c)
    b = np.where(mask, b_u, b_c)
    c = np.where(mask, c_u, c_c)
    
    # --- B. Second Membre RHS (d) selon v et T ---
    # On utilise 'w' (temps n) pour l'explicite en Y
    w_inner = w[1:-1, 1:-1]
    d = w_inner.copy()
    
    # Diffusion Y (Explicite)
    diff_y = Fy * (w[2:, 1:-1] - 2*w_inner + w[:-2, 1:-1])
    
    # Convection Y (Explicite)
    v_y = v[1:-1, 1:-1]
    Pey = np.abs(v_y) * dy / diff_coeff
    mask_y = Pey >= 2
    
    Cy = v_y * dt / (4 * dy)
    conv_y_c = Cy * (w[2:, 1:-1] - w[:-2, 1:-1]) # Centré
    
    vp, vm = np.maximum(v_y, 0), np.minimum(v_y, 0)
    Cuy = dt / (2 * dy)
    conv_y_u = Cuy * (vp*(w_inner - w[:-2, 1:-1]) + vm*(w[2:, 1:-1] - w_inner)) # Upwind
    
    conv_y = np.where(mask_y, conv_y_u, conv_y_c)
    
    # Terme Source Boussinesq (dT/dx) au temps n
    dTdx = (T[1:-1, 2:] - T[1:-1, :-2]) / (2*dx)
    source = s_half * dTdx
    
    # Assemblage RHS
    d = d + diff_y - conv_y + source
    
    # Ajout des CL Dirichlet (qui sont dans w_star depuis le pré-calcul)
    d[:, 0]  -= a[:, 0]  * w_star[1:-1, 0]
    d[:, -1] -= c[:, -1] * w_star[1:-1, -1]
    
    # --- C. Résolution Thomas ---
    # Le résultat EST w_star (intérieur)
    w_star[1:-1, 1:-1] = solve_thomas_vectorized(a, b, c, d)
    
    
    # ==========================================
    # ÉTAPE 2 : Y-IMPLICITE (Calcul de w_new)
    # ==========================================
    # On utilise w_star (fraîchement calculé) pour faire l'explicite en X
    # On transpose tout pour résoudre par colonnes
    
    w_trans = w_star.T       # (Nx, Ny)
    w_new_trans = w_new.T    # On écrira dedans
    u_trans = u.T
    v_trans = v.T
    T_trans = T.T
    
    # On met aussi à jour les CL transposées dans w_new_trans pour Thomas
    # (Les CL Haut/Bas deviennent Gauche/Droite dans la transposée)
    # w_star contient déjà les bonnes valeurs aux bords, on les copie dans w_new
    w_new_trans[:, 0] = w_trans[:, 0]   # Bas
    w_new_trans[:, -1] = w_trans[:, -1] # Haut
    
    # --- A. Matrices LHS (Y implicite devient X dans la transposée) ---
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
    b = np.where(mask_y, b_u, b_c)
    c = np.where(mask_y, c_u, c_c)
    
    # --- B. Second Membre RHS (Explicit X avec w_star) ---
    w_star_inner_T = w_trans[1:-1, 1:-1]
    d_y = w_star_inner_T.copy()
    
    # Diffusion X (Explicite sur w_star)
    diff_x = Fx * (w_trans[2:, 1:-1] - 2*w_star_inner_T + w_trans[:-2, 1:-1])
    
    # Convection X (Explicite sur w_star)
    u_x = u_trans[1:-1, 1:-1]
    Pex = np.abs(u_x) * dx / diff_coeff
    mask_x = Pex >= 2
    
    Cx = u_x * dt / (4 * dx)
    conv_x_c = Cx * (w_trans[2:, 1:-1] - w_trans[:-2, 1:-1])
    
    up, um = np.maximum(u_x, 0), np.minimum(u_x, 0)
    Cux = dt / (2 * dx)
    conv_x_u = Cux * (up*(w_star_inner_T - w_trans[:-2, 1:-1]) + um*(w_trans[2:, 1:-1] - w_star_inner_T))
    
    conv_x = np.where(mask_x, conv_x_u, conv_x_c)
    
    # Source Boussinesq (dT/dx) - Attention dx est l'axe 0 dans la transposée
    dTdx_trans = (T_trans[2:, 1:-1] - T_trans[:-2, 1:-1]) / (2*dx)
    source = s_half * dTdx_trans
    
    # Assemblage
    d_y = d_y + diff_x - conv_x + source
    
    # CL Dirichlet (depuis w_new_trans qui a les bords à jour)
    d_y[:, 0]  -= a[:, 0]  * w_new_trans[1:-1, 0]
    d_y[:, -1] -= c[:, -1] * w_new_trans[1:-1, -1]
    
    # --- C. Résolution Thomas ---
    res_y = solve_thomas_vectorized(a, b, c, d_y)
    
    # Stockage dans w_new (transposé)
    w_new_trans[1:-1, 1:-1] = res_y
    
    # Re-transposition finale vers w_new
    # (Comme w_new_trans est une vue de w_new.T, w_new est déjà à jour, 
    # mais pour être sûr de l'ordre mémoire :)
    w_new[:, :] = w_new_trans.T
    
    return w_new