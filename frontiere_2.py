import numpy as np

Nx = 81          # nombre total de points en x
Ny = 81          # nombre total de points en y
U0 = 1.0         # vitesse adimensionnée du couvercle

psi   = np.zeros((Ny, Nx))
omega = np.zeros((Ny, Nx))


def apply_psi_bc(psi):
    psi[0, :]     = 0.0      # bas      (j = 0)
    psi[-1, :]    = 0.0      # haut     (j = Ny-1)
    psi[:, 0]     = 0.0      # gauche   (i = 0)
    psi[:, -1]    = 0.0      # droite   (i = Nx-1)


def compute_omega_boundary_from_psi(
    psi: np.ndarray,
    Nx: int,
    Ny: int,
    U0: float = 1.0
) -> np.ndarray:
    """
    Compute vorticity values on the domain boundaries from streamfunction psi,
    for a 2D lid-driven cavity in nondimensional form.

    Parameters
    ----------
    psi : ndarray of shape (Ny, Nx)
        Streamfunction at time n (psi^n).
        Assumed to satisfy psi = 0 on all boundaries.
    Nx, Ny : int
        Total number of grid points in x and y directions.
    U0 : float, optional
        Nondimensional lid velocity (default = 1.0).

    Returns
    -------
    omega_bc : ndarray of shape (Ny, Nx)
        Array containing vorticity values on the boundaries.
        Interior points are set to 0 and must be filled separately
        by the transport equation (point 3 of the subject).

    Notes
    -----
    - Grid is assumed uniform on [0,1]x[0,1]
    - dx = 1/(Nx-1), dy = 1/(Ny-1)
    - Indices:
        interior : i=1..Nx-2, j=1..Ny-2
        bottom   : j=0
        top      : j=Ny-1
        left     : i=0
        right    : i=Nx-1
    """
    dx = 1.0 / (Nx - 1)
    dy = 1.0 / (Ny - 1)

    # Initialise output (only boundaries will be filled)
    omega_bc = np.zeros((Ny, Nx))

    # -----------------------
    # Bottom wall (j = 0), u = 0
    # -----------------------
    omega_bc[0, 1:Nx-1] = -2.0 * psi[1, 1:Nx-1] / (dy * dy)

    # -----------------------
    # Top wall (j = Ny-1), lid u = U0
    # -----------------------
    omega_bc[Ny-1, 1:Nx-1] = (
        -2.0 * psi[Ny-2, 1:Nx-1] / (dy * dy)
        - 2.0 * U0 / dy
    )

    # -----------------------
    # Left wall (i = 0), v = 0
    # -----------------------
    omega_bc[1:Ny-1, 0] = -2.0 * psi[1:Ny-1, 1] / (dx * dx)

    # -----------------------
    # Right wall (i = Nx-1), v = 0
    # -----------------------
    omega_bc[1:Ny-1, Nx-1] = -2.0 * psi[1:Ny-1, Nx-2] / (dx * dx)

    # -----------------------
    # Corners: average of adjacent edges
    # -----------------------
    omega_bc[0, 0] = 0.5 * (omega_bc[0, 1] + omega_bc[1, 0])
    omega_bc[0, Nx-1] = 0.5 * (omega_bc[0, Nx-2] + omega_bc[1, Nx-1])
    omega_bc[Ny-1, 0] = 0.5 * (omega_bc[Ny-1, 1] + omega_bc[Ny-2, 0])
    omega_bc[Ny-1, Nx-1] = 0.5 * (
        omega_bc[Ny-1, Nx-2] + omega_bc[Ny-2, Nx-1]
    )

    return omega_bc
