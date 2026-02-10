import numpy as np
import time
from IPython.display import clear_output

# Import des modules de calcul
from components.computation.solve_psi import solve_psi_SOR
from components.computation.solve_temperature import solve_adi_T
from components.computation.solve_vorticity import solve_adi_w
from components.computation.compute_velocity import get_velocity
from components.computation.compute_nusselt import get_average_nusselt

def global_resolution(nx, ny, dt_max, Re, Ra, direction="+", depth=0):
    # --- 1. Paramètres Adimensionnés ---
    dx = 1 / (nx - 1)
    dy = 1 / (ny - 1)
    Pr = 0.71
    
    u_lid = 1.0
    if direction == '-' or direction == "-": 
        u_lid = -1.0
        
    # Le nombre de Richardson dépend de Ra et Re
    Ri = Ra / (Pr * Re**2)
    
    print(f"--- Initialisation ---")
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
    max_iter = 40000
    
    # CRITÈRES D'ARRÊT (Absolus)
    tol_w_abs = 1e-4
    tol_T_abs = 1e-4  
    tol_Nu_abs = 5e-3

    Ri = Ra/(Pr*Re**2)

    if Ri > 1:
        tol_Nu_abs = 3e-2

    # CRITÈRES DE STAGNATION (Plateau)
    tol_plateau = 2e-2
    window_size = 40   # Fenêtre pour la moyenne glissante
    
    hist_w, hist_T, hist_Nu = [], [], []

    dt = dt_max
    target_cfl = 0.5
    
    img_dic = {'T': [], 'w': [], 'psi': []}
    history = {'res_w': [], 'res_T': [], 'res_Nu': [], 'dt': [], 'Nu_hot': [], 'Nu_cold': []}
    
    # --- 4. Boucle Temporelle ---
    start_time = time.time()
    print("Démarrage du calcul...")
    
    relaunch_simulation = False
    n = 0
    stop_simu = False
    
    while n < max_iter:
        
        # CFL Dynamique
        if n > 500 and target_cfl < 2.0:
            target_cfl = min(target_cfl * 1.01, 2.0)     
        v_max = np.max(np.sqrt(u**2 + v**2)) + 1e-9
        dt_stability = target_cfl * min(dx, dy) / v_max
        dt = min(dt * 1.01, dt_stability, dt_max)
        
        # --- RÉSOLUTION ---
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
            
            # Affichage périodique
            if n % 500 == 0:
                print(f"It {n:5d} | dt={dt:.1e} | Res_Nu={res_Nu:.1e} | Res_w={res_w:.1e} | Res_T={res_T:.1e}")

            # Sauvegarde historique
            history['res_w'].append(res_w)
            history['res_T'].append(res_T)
            history['res_Nu'].append(res_Nu)
            history['dt'].append(dt)
            history['Nu_hot'].append(Nu_h)
            history['Nu_cold'].append(Nu_c)

            # --- SÉCURITÉ : Divergence ---
            if n > 50 and (res_w > 1.5 or res_T > 1.5 or np.isnan(res_w) or res_Nu >= 1.5):
                print(f"\n=== ALERTE : Divergence détectée à It {n} ===")
                stop_simu = True
                relaunch_simulation = True
                break
            
            # --- GESTION FENÊTRE GLISSANTE ---
            hist_w.append(res_w)
            hist_T.append(res_T)
            hist_Nu.append(res_Nu) 
            
            if len(hist_w) > window_size:
                hist_w.pop(0)
                hist_T.pop(0)
                hist_Nu.pop(0)

            # --- VÉRIFICATION DES CRITÈRES D'ARRÊT ---
            if n >= 500 and len(hist_w) == window_size:
                
                # Fonction locale pour vérifier "Tolérance OU Plateau"
                def check_status(history_data, current_val, tol_abs):
                    # 1. Critère Absolu (c'est petit ?)
                    is_small = current_val < tol_abs
                    
                    # 2. Critère Plateau (ça ne bouge plus ?)
                    # On compare la moyenne de la 1ère moitié vs 2ème moitié
                    mid = len(history_data) // 2
                    mean_old = np.mean(history_data[:mid])
                    mean_new = np.mean(history_data[mid:])
                    evolution = abs(mean_new - mean_old) / (abs(mean_new) + 1e-12)
                    is_plateau = evolution < tol_plateau
                    
                    return is_small, is_plateau

                # Vérification pour chaque variable
                w_small, w_plat = check_status(hist_w, res_w, tol_w_abs)
                T_small, T_plat = check_status(hist_T, res_T, tol_T_abs)
                
                # Pour Nu, on regarde si la VALEUR (Nu_h) est stable, ou si le BILAN (res_Nu) est bon
                Nu_small, Nu_plat = check_status(hist_Nu, res_Nu, tol_Nu_abs)# On ignore le critère abs pour la valeur de Nu

                # LOGIQUE GLOBALE :
                # On arrête si TOUT le monde est content (soit petit, soit stable)
                w_ok = w_small or w_plat
                T_ok = T_small or T_plat
                Nu_ok = Nu_small or Nu_plat
                
                if w_ok and T_ok and Nu_ok:
                    print(f"\n=== CONVERGENCE ATTEINTE : It {n} ===")
                    print(f"Status w  : {'Small' if w_small else 'Plateau'}")
                    print(f"Status T  : {'Small' if T_small else 'Plateau'}")
                    print(f"Status Nu : {'Small' if Nu_small else 'Plateau'}")
                    stop_simu = True

            if stop_simu:
                print(f"Res_w: {res_w:.2e} | Res_T: {res_T:.2e} | Res_Nu (Balance): {res_Nu:.2e}")
                print(f"Nu Moyen: {(abs(Nu_h)+abs(Nu_c))/2:.4f}")
                
                img_dic['T'].append(T_computed.copy())
                img_dic['w'].append(w_computed.copy())
                img_dic['psi'].append(psi.copy())
                break

        # Mise à jour
        T[:] = T_computed[:]
        w[:] = w_computed[:]
        
        if n % 2000 == 0:
            img_dic['T'].append(T.copy())
            img_dic['w'].append(w.copy())
            img_dic['psi'].append(psi.copy())
            
        n += 1
    
    if relaunch_simulation:
        if depth >= 5:
            print(f"ABANDON : Trop de tentatives de relance (Depth={depth}).")
            return img_dic 
            
        new_dt = dt_max / 2
        print(f"\n>>> RELANCE AUTOMATIQUE ({depth+1}/5) : Nouveau dt_max = {new_dt:.1e}")
        return global_resolution(nx, ny, new_dt, Re, Ra, direction, depth=depth+1)
    
    total_time = time.time() - start_time
    print(f"Calcul terminé en {total_time:.2f} s ({n} itérations)")
    img_dic.update(history)
    
    return img_dic