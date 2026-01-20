import numpy as np

def get_average_nusselt(T, dx):
    """
    Calcule le nombre de Nusselt moyen sur la paroi chaude (gauche, x=0).
    Utilise la logique de la maille fictive (équivalent gradient ordre 1).
    """
    T_paroi = T[:, 0]
    T_1 = T[:, 1]
    
    # Nu local = (T_paroi - T_1) / dx
    nu_local = (T_paroi - T_1) / dx
    
    # Retourne la moyenne sur toute la hauteur de la paroi (axe y)
    return np.mean(nu_local)