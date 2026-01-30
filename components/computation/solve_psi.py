import numpy as np

def solve_psi_SOR(psi, w, dx, dy, alpha_sor, tol):
    """
    Résolution de l'équation de Poisson : Nabla^2 psi = -w
    Méthode : SOR (Successive Over-Relaxation) avec vectorisation Red-Black.
    """
    Ny, Nx = psi.shape
    
    # --- 1. Pré-calcul des constantes physiques ---
    beta = dx / dy
    beta_sq = beta**2
    
    # Le dénominateur commun de l'algo SOR
    # Formule : Psi_new = (1-w)Psi_old + w * (Voisins + Source) / Denom
    denom = 2 * (1 + beta_sq)
    inv_denom = 1.0 / denom
    
    # Terme source (w * dx^2) pré-calculé pour ne pas le refaire 1000 fois
    source_term_grid = w * dx**2

    # --- 2. Fonction de mise à jour locale (Closure) ---
    def compute_sor_block(psi_center, psi_left, psi_right, psi_top, psi_bottom, source_block):
        """
        Applique la formule SOR sur un bloc de la grille.
        Les variables sont explicites pour la lisibilité.
        """
        # Somme des voisins horizontaux (i-1, i+1)
        sum_x = psi_left + psi_right
        
        # Somme des voisins verticaux (j-1, j+1) pondérée par beta²
        sum_y = beta_sq * (psi_top + psi_bottom)
        
        # Formule de Gauss-Seidel avec relaxation (alpha_sor)
        # Target = (Somme_Voisins + Source) / Coeff_Central
        target = (sum_x + sum_y + source_block) * inv_denom
        
        # Combinaison linéaire : (1 - alpha) * Old + alpha * Target
        return (1 - alpha_sor) * psi_center + alpha_sor * target

    # --- 3. Boucle de Convergence ---
    max_iter = 2000
    for k in range(max_iter):
        psi_prev = psi.copy()
        
        # === ÉTAPE 1 : CASES "ROUGES" (Somme des indices i+j paire) ===
        # Motif A : Lignes impaires (1, 3...), Colonnes impaires (1, 3...)
        # Indices Python : [1::2]
        psi[1:-1:2, 1:-1:2] = compute_sor_block(
            psi_center = psi[1:-1:2, 1:-1:2],
            psi_left   = psi[1:-1:2, 0:-2:2],  # Voisin Gauche
            psi_right  = psi[1:-1:2, 2::2],    # Voisin Droite
            psi_top    = psi[0:-2:2, 1:-1:2],  # Voisin Haut (Indice j-1)
            psi_bottom = psi[2::2,   1:-1:2],  # Voisin Bas  (Indice j+1)
            source_block = source_term_grid[1:-1:2, 1:-1:2]
        )
        
        # Motif B : Lignes paires (2, 4...), Colonnes paires (2, 4...)
        psi[2:-1:2, 2:-1:2] = compute_sor_block(
            psi_center = psi[2:-1:2, 2:-1:2],
            psi_left   = psi[2:-1:2, 1:-3:2],
            psi_right  = psi[2:-1:2, 3::2],
            psi_top    = psi[1:-3:2, 2:-1:2],
            psi_bottom = psi[3::2,   2:-1:2],
            source_block = source_term_grid[2:-1:2, 2:-1:2]
        )

        # === ÉTAPE 2 : CASES "NOIRES" (Somme des indices i+j impaire) ===
        # Motif C : Lignes impaires (1, 3...), Colonnes paires (2, 4...)
        psi[1:-1:2, 2:-1:2] = compute_sor_block(
            psi_center = psi[1:-1:2, 2:-1:2],
            psi_left   = psi[1:-1:2, 1:-3:2],
            psi_right  = psi[1:-1:2, 3::2],
            psi_top    = psi[0:-2:2, 2:-1:2],
            psi_bottom = psi[2::2,   2:-1:2],
            source_block = source_term_grid[1:-1:2, 2:-1:2]
        )

        # Motif D : Lignes paires (2, 4...), Colonnes impaires (1, 3...)
        psi[2:-1:2, 1:-1:2] = compute_sor_block(
            psi_center = psi[2:-1:2, 1:-1:2],
            psi_left   = psi[2:-1:2, 0:-2:2],
            psi_right  = psi[2:-1:2, 2::2],
            psi_top    = psi[1:-3:2, 1:-1:2],
            psi_bottom = psi[3::2,   1:-1:2],
            source_block = source_term_grid[2:-1:2, 1:-1:2]
        )
        
        # --- 4. Vérification de la convergence (tous les 50 pas) ---
        if k % 50 == 0:
            diff = np.max(np.abs(psi - psi_prev))
            if diff < tol:
                return psi
                
    return psi