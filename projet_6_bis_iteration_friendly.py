import numpy as np
import time
from IPython.display import clear_output

# Import des modules de calcul
from components.computation.solve_psi import solve_psi_SOR
from components.computation.solve_temperature import solve_adi_T
from components.computation.solve_vorticity import solve_adi_w
from components.computation.compute_velocity import get_velocity
from components.computation.compute_nusselt import get_average_nusselt

def global_resolution(nx, ny, dt_max, Re, Ra, direction="+"):
    """
    Solveur Adimensionné (u_lid = 1.0 ou -1.0).
    L'argument 'nu' n'est pas utilisé pour le calcul (piloté par Re), 
    il est gardé juste pour compatibilité avec tes anciens appels.
    """
    # --- 1. Paramètres Adimensionnés ---
    dx = 1 / (nx - 1)
    dy = 1 / (ny - 1)
    Pr = 0.71
    
    u_lid = 1.0
    if direction == '-': 
        u_lid = -1.0
        
    # Le nombre de Richardson dépend de Ra et Re
    Ri = Ra / (Pr * Re**2)
    
    print(f"--- Initialisation (Adimensionnée) ---")
    print(f"Grid: {nx}x{ny} | Re: {Re} | Ra: {Ra:.1e} | Ri: {Ri:.2f}")
    print(f"Direction Paroi {np.sign(u_lid)}")

    # --- 2. Allocation Mémoire ---
    T = np.zeros((ny, nx))
    w = np.zeros((ny, nx))
    psi = np.zeros((ny, nx))
    
    # On initialise la vitesse avec la bonne condition au bord (u_lid)
    u, v = get_velocity(psi, dx, dy, u_lid)    
    T[:, 0] = 1.0 
    
    # Buffers
    T_buffer = {'T_star': T.copy(), 'T_new': T.copy()}
    w_buffer = {'w_star': np.zeros((ny, nx)), 'w_new': np.zeros((ny, nx))}

    # --- 3. Paramètres Numériques ---
    alpha_sor = 1.725
    tol_sor = 1e-4
    max_iter = 100000
    
    # CRITÈRES D'ARRÊT
    tol_w_abs = 1e-4
    tol_T_abs = 1e-4   
    tol_Nu_abs = 1e-2   

    tol_stagnation = 1e-2  
    safety_threshold = 1e-3 
    
    window_size = 100 
    hist_w, hist_T, hist_Nu = [], [], []

    dt = dt_max
    target_cfl = 0.5 if Ra <= 1e5 else 0.3
    
    img_dic = {'T': [], 'w': [], 'psi': [], 'u': [], 'v': []}
    history = {'res_w': [], 'res_T': [], 'res_Nu': [], 'dt': [], 'Nu_hot': [], 'Nu_cold': []}
    
    # --- 4. Boucle Temporelle ---
    start_time = time.time()
    print("Démarrage du calcul...")
    
    n = 0
    stop_simu = False
    while n < max_iter:
        
        # CFL Dynamique
        if n > 500 and target_cfl < 2.0:
            target_cfl = min(target_cfl * 1.02, 2.0)     
        v_max = np.max(np.sqrt(u**2 + v**2)) + 1e-9
        dt_stability = target_cfl * min(dx, dy) / v_max
        dt = min(dt * 1.05, dt_stability, dt_max)
        
        # --- RÉSOLUTION ---
        # Note : On passe bien u_lid (1.0 ou -1.0) aux fonctions
        T_computed = solve_adi_T(T, u, v, 1/(Re*Pr), dt, dx, dy, work_buffer=T_buffer)
        w_computed = solve_adi_w(w, T_computed, psi, u, v, 1/Re, Ri, dt, dx, dy, u_lid, work_buffer=w_buffer)
        psi = solve_psi_SOR(psi, w_computed, dx, dy, alpha_sor, tol_sor)
        get_velocity(psi, dx, dy, u_lid, out_u=u, out_v=v)
        
        # --- CONTRÔLE (Toutes les 50 itérations) ---
        if n % 50 == 0:
            scale_w = np.max(np.abs(w_computed)) + 1e-6
            scale_T = np.max(np.abs(T_computed)) + 1e-6
            res_w = np.max(np.abs(w_computed - w)) / scale_w
            res_T = np.max(np.abs(T_computed - T)) / scale_T
            
            Nu_h, Nu_c = get_average_nusselt(T_computed, dx)
            res_Nu = abs(Nu_h - Nu_c) / (abs(Nu_h) + 1e-9)
            
            if n % 500 == 0:
                print(f"It {n:5d} | dt={dt:.1e} | Res_Nu={res_Nu:.1e} | Res_w={res_w:.1e} | Res_T={res_T:.1e}")

            history['res_w'].append(res_w)
            history['res_T'].append(res_T)
            history['res_Nu'].append(res_Nu)
            history['dt'].append(dt)
            history['Nu_hot'].append(Nu_h)
            history['Nu_cold'].append(Nu_c)
            
            hist_w.append(res_w)
            hist_T.append(res_T)
            hist_Nu.append(res_Nu)
            
            if len(hist_w) > window_size:
                hist_w.pop(0)
                hist_T.pop(0)
                hist_Nu.pop(0)

            # --- CONDITION D'ARRÊT ---
            mean_w = np.mean(hist_w)
            mean_T = np.mean(hist_T)
            mean_Nu = np.mean(hist_Nu)
            
            diff_rel_w = abs(res_w - mean_w) / (mean_w + 1e-12)
            diff_rel_T = abs(res_T - mean_T) / (mean_T + 1e-12)
            diff_rel_Nu = abs(res_Nu - mean_Nu) / (mean_Nu + 1e-12)
            
            w_ok = (res_w < tol_w_abs) or (diff_rel_w < tol_stagnation and res_w < safety_threshold)
            T_ok = (res_T < tol_T_abs) or (diff_rel_T < tol_stagnation and res_T < safety_threshold)
            Nu_ok = (res_Nu < tol_Nu_abs) or (diff_rel_Nu < tol_stagnation)
            
            if n >= 500 and len(hist_w) == window_size and w_ok and T_ok and Nu_ok :
                stop_simu = True
            
            if stop_simu:
                print(f"\n=== ARRÊT : It {n} ===")
                print(f"Res_w: {res_w:.2e} | Res_T: {res_T:.2e} | Res_Nu (Balance): {res_Nu:.2e}")
                print(f"Nu Moyen: {(abs(Nu_h)+abs(Nu_c))/2:.4f}")
                
                img_dic['T'].append(T_computed.copy())
                img_dic['w'].append(w_computed.copy())
                img_dic['psi'].append(psi.copy())
                img_dic['u'].append(u.copy())
                img_dic['v'].append(v.copy())
                break

        # Mise à jour
        T[:] = T_computed[:]
        w[:] = w_computed[:]
        
        if n % 2000 == 0:
            img_dic['T'].append(T.copy())
            img_dic['w'].append(w.copy())
            img_dic['psi'].append(psi.copy())
            
        n += 1
        
    total_time = time.time() - start_time
    print(f"Calcul terminé en {total_time:.2f} s ({n} itérations)")
    img_dic.update(history)
    
    # On renvoie u_lid (qui vaut +/- 1) au lieu de l'ancien U0
    return img_dic