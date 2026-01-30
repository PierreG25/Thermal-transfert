import numpy as np
import time
from IPython.display import clear_output

# Import des modules de calcul (supposés être dans components/computation/)
# Assurez-vous que ces fichiers contiennent les versions VECTORISÉES données précédemment.
from components.computation.solve_psi import solve_psi_SOR
from components.computation.solve_temperature import solve_adi_T
from components.computation.solve_vorticity import solve_adi_w
from components.computation.compute_velocity import get_velocity
from components.computation.compute_nusselt import get_average_nusselt

def global_resolution(nx, ny, Lx, Ly, dt_init, nu, Re, Ra, direction="+"):
    """
    Solveur principal optimisé (Mémoire & Vectorisation).
    """
    # --- 1. Paramètres Géométriques & Physiques ---
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    Pr = 0.71
    
    # Vitesse caractéristique U0 (Basée sur Re = U0 * L / nu)
    U0 = Re * nu / Lx
    
    # Gestion de la direction du couvercle
    # "+" : Aiding Flow (Si gauche chaud, droite froid -> Rotation horaire naturelle)
    # "-" : Opposing Flow (Couvercle va à gauche, contre la convection naturelle)
    if direction == '-':
        U0 = -U0
        
    # Nombre de Richardson (Couplage Convection Naturelle / Forcée)
    # Ri = Gr / Re^2
    Ri = Ra / (Pr * Re**2)
    
    print(f"--- Initialisation ---")
    print(f"Grid: {nx}x{ny} | Re: {Re} | Ra: {Ra:.1e} | Ri: {Ri:.2f}")
    print(f"U0: {U0:.4f} m/s | dx: {dx:.2e}")

    # --- 2. Allocation Mémoire (Buffers) ---
    # On crée tous les tableaux MAINTENANT pour ne pas le faire dans la boucle (Gain vitesse)
    
    # Champs principaux
    T = np.zeros((ny, nx))
    w = np.zeros((ny, nx))
    psi = np.zeros((ny, nx))
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    
    # Condition Initiale Température (Paroi Chaude à Gauche)
    T[:, 0] = 1.0 
    
    # Buffers de travail pour ADI (évite allocation dynamique)
    # T_new et w_new servent de stockage temporaire pour le pas n+1
    T_buffer = {
        'T_star': np.zeros((ny, nx)), 
        'T_new': np.zeros((ny, nx))
    }
    # Important : Initialiser les buffers avec l'état initial
    T_buffer['T_new'][:, 0] = 1.0 
    T_buffer['T_star'][:, 0] = 1.0

    w_buffer = {
        'w_star': np.zeros((ny, nx)), 
        'w_new': np.zeros((ny, nx))
    }

    # --- 3. Paramètres Numériques ---
    alpha_sor = 1.8         # Relaxation SOR (1.7 à 1.9 est optimal pour nx=51-100)
    tol_sor = 1e-4          # Précision Poisson
    tol_steady = 1e-5       # Convergence stationnaire globale
    tol_Nu = 5e-3
    max_iter = 100000       # Sécurité
    
    # Gestion du Pas de Temps (Adaptatif)
    dt = dt_init            # Valeur courante
    target_cfl = 0.5        # CFL cible (prudent au début)
    if Ra > 1e5: target_cfl = 0.3 # Plus prudent si très turbulent
    
    # Stockage résultats
    img_dic = {'T': [], 'w': [], 'psi': [], 'u': [], 'v': []}
    history = {'res_w': [], 'res_T': [], 'res_Nu': [], 'dt': [], 
           'Nu_hot': [], 'Nu_cold': []}
    

    # --- 4. Boucle Temporelle ---
    start_time = time.time()
    print("Démarrage du calcul...")
    
    n = 0
    while n < max_iter:
        
        # --- AJOUT : DÉBRIDAGE PROGRESSIF DU CFL ---
        # Après 500 itérations, on augmente doucement le CFL cible jusqu'à 2.0
        if n > 500 and target_cfl < 2.0:
            target_cfl = min(target_cfl * 1.02, 2.0)
            
        # A. Adaptative Time Stepping (Calcul standard)
        v_max = np.max(np.sqrt(u**2 + v**2)) + 1e-9
        dt_stability = target_cfl * min(dx, dy) / v_max
        
        # On ne dépasse pas dt_init (qui sert de plafond absolu)
        dt = min(dt * 1.05, dt_stability, dt_init)        
        # A. Adaptative Time Stepping (CFL)
        # On calcule la vitesse max pour ajuster dt
        # Vmax ne doit pas être 0 pour éviter division par zéro
        v_max = np.max(np.sqrt(u**2 + v**2)) + 1e-9
        
        # CFL = V * dt / dx => dt_max = CFL * dx / V
        dt_stability = target_cfl * min(dx, dy) / v_max
        
        # On lisse l'évolution de dt (ne pas changer trop brusquement)
        # On ne dépasse pas dt_init (qui est le dt max souhaité par l'utilisateur)
        dt = min(dt * 1.05, dt_stability, dt_init)
        
        T_computed = solve_adi_T(T, u, v, 1/(Re*Pr), dt, dx, dy, work_buffer=T_buffer)
        
        w_computed = solve_adi_w(w, T_computed, psi, u, v, 1/Re, Ri, dt, dx, dy, U0, work_buffer=w_buffer)
        
        psi = solve_psi_SOR(psi, w_computed, dx, dy, alpha_sor, tol_sor)
        
        # Mise à jour "In-Place" dans u et v pour économiser mémoire
        get_velocity(psi, dx, dy, U0, out_u=u, out_v=v)
        
        # C. Suivi Convergence (tous les 100 itérations pour gagner du temps CPU)
        if n % 50 == 0:
            # Calcul des résidus (différence relative normalisée)
            scale_w = np.max(np.abs(w_computed)) + 1e-6
            scale_T = np.max(np.abs(T_computed)) + 1e-6
            
            res_w = np.max(np.abs(w_computed - w)) / scale_w
            res_T = np.max(np.abs(T_computed - T)) / scale_T
            
            # Nusselt
            Nu_h, Nu_c = get_average_nusselt(T_computed, dx)
            if abs(Nu_h) < 1e-9: Nu_h = 1e-9
            res_Nu = abs(Nu_h - Nu_c) / abs(Nu_h)
            
            # Stockage
            history['res_w'].append(res_w)
            history['res_T'].append(res_T)
            history['res_Nu'].append(res_Nu)
            history['dt'].append(dt)
            history['Nu_hot'].append(Nu_h)
            history['Nu_cold'].append(Nu_c)
            
            # Affichage console
            if n % 500 == 0:
                print(f"It {n:5d} | dt={dt:.1e} | Res_Nu={res_Nu:.1e} | Res_w={res_w:.1e} | Res_T={res_T:.1e} | Nu={abs(Nu_h):.3f}")

            # Critère d'arrêt (Convergence Stationnaire)
            # On demande que T et w soient stables, ET que le bilan énergétique (Nu) soit bon (<1%)
            if n > 500 and res_w < tol_steady and res_T < tol_steady and res_Nu < tol_Nu:
                print(f"\n=== CONVERGENCE ATTEINTE (It {n}) ===")
                print(f"Nu Moyen: {(abs(Nu_h)+abs(Nu_c))/2:.4f}")
                # Sauvegarde finale
                img_dic['T'].append(T_computed.copy())
                img_dic['w'].append(w_computed.copy())
                img_dic['psi'].append(psi.copy())
                img_dic['u'].append(u.copy())
                img_dic['v'].append(v.copy())
                break
        
        # D. Mise à jour des états pour n+1
        # Copie rapide des valeurs du buffer vers les tableaux principaux
        T[:] = T_computed[:]
        w[:] = w_computed[:]
        
        # Sauvegarde périodique pour animation (tous les 2000 pas)
        if n % 2000 == 0:
            img_dic['T'].append(T.copy())
            img_dic['w'].append(w.copy())
            img_dic['psi'].append(psi.copy())
            
        n += 1
        
    # --- 5. Finalisation ---
    total_time = time.time() - start_time
    print(f"Calcul terminé en {total_time:.2f} s ({n} itérations)")
    
    # Ajout des historiques au dictionnaire de retour
    img_dic.update(history)
    
    # On retourne U0 (calculé) et le dictionnaire de résultats
    return U0, img_dic