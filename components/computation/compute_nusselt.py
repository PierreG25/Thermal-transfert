import numpy as np

def get_average_nusselt(T, dx):
    """
    Calcule le nombre de Nusselt moyen avec une précision d'ordre 2.
    CORRIGÉ : Signes ajustés pour avoir des valeurs positives des deux côtés.
    """
    # 1. Paroi chaude (x=0)
    # Gradient au bord (Différence décentrée)
    grad_hot = (-3*T[:, 0] + 4*T[:, 1] - T[:, 2]) / (2*dx)
    nu_hot_local = -grad_hot 
    
    # 2. Paroi froide (x=Lx)
    # Gradient au bord (Différence décentrée)
    grad_cold = (3*T[:, -1] - 4*T[:, -2] + T[:, -3]) / (2*dx)
    nu_cold_local = -grad_cold 
    
    return np.mean(nu_hot_local), np.mean(nu_cold_local)