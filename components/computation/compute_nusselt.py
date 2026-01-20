import numpy as np

def get_average_nusselt(T, dx):
    """
    Calcule le nombre de Nusselt moyen sur la paroi chaude (gauche) 
    et sur la paroi froide (droite).
    """
    # 1. Paroi chaude (x=0, i=0) : T=1.0
    # Le gradient est approx (T[i=1] - T[i=0]) / dx
    # Nu_hot = -dT/dx = (T[i=0] - T[i=1]) / dx
    nu_hot_local = (T[:, 0] - T[:, 1]) / dx
    nu_hot_avg = np.mean(nu_hot_local)
    
    # 2. Paroi froide (x=Lx, i=Nx-1) : T=0.0
    # Le gradient est approx (T[i=Nx-1] - T[i=Nx-2]) / dx
    # Nu_cold = -dT/dx = (T[i=Nx-2] - T[i=Nx-1]) / dx
    nu_cold_local = (T[:, -2] - T[:, -1]) / dx
    nu_cold_avg = np.mean(nu_cold_local)
    
    return nu_hot_avg, nu_cold_avg