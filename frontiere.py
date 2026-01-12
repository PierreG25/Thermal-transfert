import numpy as np

Nx = 81          # nombre total de points en x
Ny = 81          # nombre total de points en y
U0 = 1.0         # vitesse adimensionnée du couvercle

dx = 1.0 / (Nx - 1)
dy = 1.0 / (Ny - 1)

psi   = np.zeros((Ny, Nx))
omega = np.zeros((Ny, Nx))


def apply_psi_bc(psi):
    psi[0, :]     = 0.0      # bas      (j = 0)
    psi[-1, :]    = 0.0      # haut     (j = Ny-1)
    psi[:, 0]     = 0.0      # gauche   (i = 0)
    psi[:, -1]    = 0.0      # droite   (i = Nx-1)


def compute_omega_boundary(omega, psi, Nx, Ny, U0):
    """
    Calcul de la vorticité aux frontières (omega^{n+1})
    à partir de psi^n, domaine adimensionné [0,1]x[0,1]
    
    psi[j,i], omega[j,i]
    """
    dx = 1.0 / (Nx - 1)
    dy = 1.0 / (Ny - 1)

    # -----------------------
    # Bas (j = 0) : paroi fixe
    # -----------------------
    # i = 1 .. Nx-2
    omega[0, 1:Nx-1] = -2.0 * psi[1, 1:Nx-1] / (dy*dy)

    # -----------------------
    # Haut (j = Ny-1) : couvercle u = U0
    # -----------------------
    omega[Ny-1, 1:Nx-1] = (
        -2.0 * psi[Ny-2, 1:Nx-1] / (dy*dy)
        - 2.0 * U0 / dy
    )

    # -----------------------
    # Gauche (i = 0) : paroi fixe
    # -----------------------
    # j = 1 .. Ny-2
    omega[1:Ny-1, 0] = -2.0 * psi[1:Ny-1, 1] / (dx*dx)

    # -----------------------
    # Droite (i = Nx-1) : paroi fixe
    # -----------------------
    omega[1:Ny-1, Nx-1] = -2.0 * psi[1:Ny-1, Nx-2] / (dx*dx)

    # -----------------------
    # Coins : moyenne (robuste)
    # -----------------------
    omega[0, 0] = 0.5 * (omega[0, 1] + omega[1, 0])
    omega[0, Nx-1] = 0.5 * (omega[0, Nx-2] + omega[1, Nx-1])
    omega[Ny-1, 0] = 0.5 * (omega[Ny-1, 1] + omega[Ny-2, 0])
    omega[Ny-1, Nx-1] = 0.5 * (omega[Ny-1, Nx-2] + omega[Ny-2, Nx-1])
