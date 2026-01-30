import numpy as np

def get_average_nusselt(T, dx):
    """
    Calcule le nombre de Nusselt moyen avec une précision d'ordre 2.
    CORRIGÉ : Signes ajustés pour avoir des valeurs positives des deux côtés.
    """
    # 1. Paroi chaude (x=0)
    # Gradient au bord (Forward Difference)
    grad_hot = (-3*T[:, 0] + 4*T[:, 1] - T[:, 2]) / (2*dx)
    # Le gradient est négatif (la température baisse), on veut un Nu positif
    nu_hot_local = -grad_hot 
    
    # 2. Paroi froide (x=Lx)
    # Gradient au bord (Backward Difference)
    grad_cold = (3*T[:, -1] - 4*T[:, -2] + T[:, -3]) / (2*dx)
    # Le gradient est négatif (ça baisse vers 0), on veut un Nu positif
    # --- CORRECTION ICI : AJOUT DU MOINS ---
    nu_cold_local = -grad_cold 
    
    return np.mean(nu_hot_local), np.mean(nu_cold_local)