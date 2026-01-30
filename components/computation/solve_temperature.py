import numpy as np
from components.computation.thomas_algorithm import solve_thomas_vectorized

def solve_adi_T(T, u, v, diff_coeff, dt, dx, dy, work_buffer=None):
    """Version vectorisée optimisée de ADI Température"""

# --- GESTION MÉMOIRE ---
    if work_buffer is not None:
        # On récupère les tableaux existants (déjà alloués)
        T_new = work_buffer['T_new']
        T_star = work_buffer['T_star']
        # On s'assure qu'ils sont propres (optionnel selon la logique, mais sûr)
        # Pas besoin de .fill(0) car on va tout écraser, mais T_star doit être copie de T aux bords
    else:
        # Comportement par défaut (plus lent)
        T_new = np.zeros_like(T)
        T_star = np.zeros_like(T)

    # ... Le reste du code est IDENTIQUE à la version vectorisée précédente ...
    # ... MAIS attention à une chose : T_star doit être initialisé correctement ...
    
    # Initialisation de T_star (copie de T nécessaire pour les itérations)
    # On peut copier tout T dans T_star pour commencer, ou juste les bords selon le besoin
    # Pour la vectorisation ADI que je t'ai donnée, on avait besoin des bords de T_star
    T_star[:] = T[:] 
    
    # Coefficients constants
    Fx = (diff_coeff * dt) / (2 * dx**2)
    Fy = (diff_coeff * dt) / (2 * dy**2)    
    # ===================================================
    # ÉTAPE 1 : X-Implicite (Résolution par lignes)
    # ===================================================
    # On travaille sur l'intérieur [1:-1]
    # u_inner a la forme (Ny, Nx-2). On inclut les bords j=0 et j=Ny-1 pour simplifier
    
    # 1. Préparation des coefficients pour TOUT le domaine (vectoriel)
    # u sur les faces concernées (toutes les lignes, colonnes intérieures)
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
    
    # Sélection Vectorisée (Hybride)
    # a, b, c sont de taille (Ny, Nx-2)
    a = np.where(mask, a_u, a_c)
    b = np.where(mask, b_u, b_c)
    c = np.where(mask, c_u, c_c)
    
    # 2. Second membre d (Explicite en Y)
    # On a besoin des dérivées en Y pour le second membre.
    # Pour vectoriser, on gère les intérieurs j=1..Ny-2
    # Les bords j=0 et j=Ny-1 sont Dirichlet (T fixe ou adiabatique traité à part)
    # Pour simplifier la vectorisation globale, on calcule tout et on corrige les bords ensuite.
    
    # Terme diffusion Y (explicite) : T_yy
    # T[j+1] - 2T[j] + T[j-1]  -> On utilise slicing (2:, :) - 2*(1:-1, :) + (:-2, :)
    # Attention: T est complet (Ny, Nx). On veut le résultat sur l'intérieur des X (:, 1:-1)
    # Slicing vertical : on prend de j=1 à Ny-2
    
    # On initialise d avec la valeur au temps n
    d = T[:, 1:-1].copy()
    
    # Diffusion Y (partout où c'est possible)
    # On crée une version décalée de T pour vectoriser T(j+1), T(j-1)
    # T_up = T[2:, 1:-1], T_down = T[:-2, 1:-1], T_center = T[1:-1, 1:-1]
    diff_y = Fy * (T[2:, 1:-1] - 2*T[1:-1, 1:-1] + T[:-2, 1:-1])
    
    # Convection Y (Hybride)
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
    
    # Application sur d (uniquement pour j=1..Ny-2)
    d[1:-1, :] = d[1:-1, :] + diff_y - conv_y
    
    # Gestion des bords Y (j=0 et j=Ny-1) - Conditions Adiabatiques ou Fixes
    # Dans ton code original, tu avais des cas spécifiques.
    # Supposons T fixe en haut/bas (sinon adapter comme avant). 
    # Pour simplifier ici : on ne résout pas l'équation sur j=0 et j=Ny-1 (Dirichlet), 
    # ou on adapte d pour Neumann.
    # Reprenons ta logique "d = T + Fy(2T...)" pour les bords.
    d[0, :]  = T[0, 1:-1]  + Fy*(2*T[1, 1:-1] - 2*T[0, 1:-1]) # Ex: adiabatique
    d[-1, :] = T[-1, 1:-1] + Fy*(2*T[-2, 1:-1] - 2*T[-1, 1:-1])
    
    # 3. Injection Conditions Limites X (Dirichlet T_star)
    # T_star aux bords x=0 et x=Nx-1 est supposé connu (ex: T[:,0]=1)
    # a[:, 0] multiplie le terme à gauche (x=0).
    d[:, 0]  -= a[:, 0] * T[:, 0] 
    # c[:, -1] multiplie le terme à droite (x=Nx-1).
    d[:, -1] -= c[:, -1] * T[:, -1]
    
    # 4. Résolution Thomas Vectorisée
    # On résout Ny systèmes de taille Nx-2
    res = solve_thomas_vectorized(a, b, c, d)
    
    # Reconstruction de T_star
    T_star[:, 0] = T[:, 0]    # BC Gauche
    T_star[:, -1] = T[:, -1]  # BC Droite
    T_star[:, 1:-1] = res     # Intérieur calculé
    
    
    # ===================================================
    # ÉTAPE 2 : Y-Implicite (Résolution par colonnes)
    # ===================================================
    # Astuce : On transpose tout pour se ramener au cas précédent
    # T_star est (Ny, Nx). On veut résoudre pour chaque i.
    # On transpose -> (Nx, Ny). Les colonnes deviennent des lignes.
    
    T_trans = T_star.T       # (Nx, Ny)
    u_trans = u.T            # (Nx, Ny)
    v_trans = v.T            # (Nx, Ny)
    
    # On travaille sur l'intérieur Y (donc indices 1:-1 dans la version transposée)
    # Matrices de taille (Nx, Ny-2)
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
    
    # Gestion Neumann (Adiabatique) aux bords Y (Haut/Bas)
    # Dans la transposée, le bord "gauche" est j=0 (Bas), "droite" est j=Ny-1 (Haut)
    # Tes conditions: a[0] += a_fictif, c[-1] += c_fictif
    # Vectorisation : on ajoute aux vecteurs a[:,0] et c[:,-1]
    # Note : Ne pas oublier que a[:,0] correspond à l'interaction avec le bord 0
    # Dans Thomas, a[:,0] n'est pas utilisé mathématiquement, MAIS
    # pour Neumann, on modifie b[:,0] et d[:,0] souvent, ou on ajoute au c.
    # Reprenons ta logique : c[0] += a[0] (réflexion).
    
    # Correction Neumann Vectorisée (T_y = 0 => T_-1 = T_1)
    # Cela revient à ajouter le coeff 'a' (terme gauche) au coeff 'c' (terme droite)
    # ou modifier b.
    # Ta méthode : c_y[0] += a_y[0] (car T_0 connecte à T_-1 qui vaut T_1)
    c[:, 0]  += a[:, 0]
    a[:, -1] += c[:, -1]
    
    # Second membre d (Explicite X Hybride)
    # On utilise T_trans (qui est T_star transposé)
    # On calcule la convection/diff X sur l'intérieur transposé (donc i=1..Nx-2)
    # Slicing sur l'axe 0 (qui est l'axe i original)
    
    # Initialisation avec T_star
    d_y = T_trans[:, 1:-1].copy()
    
    # Diffusion X explicite (sur les indices intérieurs du tableau transposé)
    # T_trans[i] correspond à la colonne i.
    # On a besoin des voisins i+1 et i-1.
    # T_right = T_trans[2:, :], T_left = T_trans[:-2, :]
    # Mais ici on résout "par colonne", donc pour chaque ligne de la transposée (chaque i),
    # on a besoin des voisins en i pour le terme explicite.
    
    # Attention, ici c'est subtil.
    # d_y est de taille (Nx, Ny-2).
    # Pour le terme explicite en X, il nous faut T(i+1), T(i-1).
    # On ne peut pas vectoriser simplement l'axe 0 de d_y car il dépend des voisins de l'axe 0.
    # MAIS, on calcule d_y pour TOUS les i (1..Nx-2).
    
    # On extrait le bloc central complet pour calculs
    # T_star_in = T_trans (Nx, Ny).
    # On veut calculer le terme source X pour i=1..Nx-2.
    # On doit restreindre d_y aux i=1..Nx-2 pour le calcul, puis injecter.
    
    # Calculons le terme explicite X pour TOUT le bloc intérieur (i=1..Nx-2, j=1..Ny-2)
    # T_trans_center = T_trans[1:-1, 1:-1]
    # T_trans_ip1    = T_trans[2:, 1:-1]  (i+1)
    # T_trans_im1    = T_trans[:-2, 1:-1] (i-1)
    
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
    
    # Mise à jour de d_y (seulement pour les i intérieurs 1..Nx-2)
    # d_y actuel contient T_star.
    # d_y est (Nx, Ny-2). On modifie les lignes 1:-1
    d_y[1:-1, :] = d_y[1:-1, :] + diff_x - conv_x
    
    # Résolution
    # a, b, c sont (Nx, Ny-2).
    # d_y est (Nx, Ny-2).
    # Mais attention, le calcul explicite X ci-dessus n'est valide que pour i=1..Nx-2.
    # Les lignes i=0 et i=Nx-1 de d_y sont les parois gauche/droite (Dirichlet).
    # Elles ne changent pas (ou sont imposées).
    # Comme on résout un système tridiagonal sur l'axe Y (colonnes), chaque "ligne" de la transposée est indépendante.
    # Les lignes i=0 et i=Nx-1 sont des conditions limites du problème global, T est fixé.
    # On peut résoudre Thomas sur tout, à condition que d_y[0] et d_y[-1] soient corrects.
    # Pour i=0 (gauche), T est imposé (1.0). On n'a pas besoin de résoudre l'ADI dessus.
    # On peut restreindre le domaine de résolution Thomas aux i=1..Nx-2.
    
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