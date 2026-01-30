import numpy as np

def get_velocity(psi, dx, dy, u_top, out_u=None, out_v=None):
    """
    Calcule les composantes de vitesse u, v à partir de psi.
    Args optionnels out_u, out_v : Tableaux existants pour éviter l'allocation mémoire.
    """
    # Si aucun buffer n'est fourni, on en crée de nouveaux (comportement par défaut)
    if out_u is None:
        out_u = np.zeros_like(psi)
    if out_v is None:
        out_v = np.zeros_like(psi)
    
    # Reset des tableaux si on les réutilise (sécurité pour les bords)
    out_u.fill(0.0)
    out_v.fill(0.0)

    # --- Calcul Vectorisé (Différences Centrées) ---
    # u = d_psi / dy
    out_u[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2*dy)
    
    # v = -d_psi / dx
    out_v[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) / (2*dx)
    
    # --- Conditions aux Limites ---
    # Couvercle entraîné (Haut)
    out_u[-1, :] = u_top
    
    # Les autres parois sont immobiles (u=0, v=0) -> Déjà fait par .fill(0.0)
    
    return out_u, out_v