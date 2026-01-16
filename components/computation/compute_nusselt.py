import numpy as np

def get_average_nusselt(T, dx):
    """
    Calcule le nombre de Nusselt moyen sur la paroi chaude (gauche, x=0).
    Utilise un schéma décentré d'ordre 2 pour le gradient.
    """
    # Nu local sur chaque point j de la paroi x=0 (colonne 0)
    # T[:, 0] est la paroi (T=1), T[:, 1] et T[:, 2] sont les premiers points intérieurs
    nu_local = -( -3*T[:, 0] + 4*T[:, 1] - T[:, 2] ) / (2 * dx)
    
    return np.mean(nu_local)