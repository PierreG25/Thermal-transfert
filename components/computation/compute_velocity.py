import numpy as np

def get_velocity(psi, dx, dy, U0):
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    # u = d_psi / dy (axe 0)
    u[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2*dy)
    # v = -d_psi / dx (axe 1)
    v[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) / (2*dx)
    
    u[-1, :] = U0 # Condition limite couvercle
    return u, v