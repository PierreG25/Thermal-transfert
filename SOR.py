import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import matplotlib as mpl

def solve_poisson_psi(psi, w, dx, dy, alpha_sor, tol):
    Ny, Nx = psi.shape
    for _ in range(500):
        calcul_matrice_psi(psi, w, dx, dy, alpha_sor,Ny,Nx)

        last_res = residu_SOR(psi, w, dx, dy, Ny, Nx)

        if last_res < tol:
            break
    assert last_res < tol, "La méthode SOR n'a pas convergé"
    return psi

def residu_SOR(psi, w, dx, dy, Ny, Nx):
    mat_residu = np.zeros_like(psi)

    mat_residu[1:Ny-1, 1:Nx-1] = (
        (psi[1:Ny-1, 2:Nx] - 2.0*psi[1:Ny-1, 1:Nx-1] + psi[1:Ny-1, 0:Nx-2]) * (dy**2)
        + (psi[2:Ny, 1:Nx-1] - 2.0*psi[1:Ny-1, 1:Nx-1] + psi[0:Ny-2, 1:Nx-1]) * (dx**2)
        + (dx**2 * dy**2) * w[1:Ny-1, 1:Nx-1]
    )

    return np.max(np.abs(mat_residu[1:Ny-1, 1:Nx-1]))

def calcul_matrice_psi(psi, w, dx, dy, alpha_sor, Ny, Nx):
    for j in range(1, Ny-1):
            for i in range(1, Nx-1):
                target = ( (psi[j, i+1] + psi[j, i-1])*dy**2 + (psi[j+1, i] + psi[j-1, i])*dx**2 + (dx**2 * dy**2)*w[j, i] ) / (2*(dx**2 + dy**2))
                psi[j, i] = (1 - alpha_sor) * psi[j, i] + alpha_sor * target
    return psi 
