import numpy as np

def solve_thomas_vectorized(a, b, c, d):
    """
    Résout M systèmes tridiagonaux de taille N simultanément.
    Args:
        a, b, c, d : Tableaux (M, N)
                     a = diagonale inférieure (Attention: a[:, 0] n'est pas utilisé)
                     b = diagonale principale
                     c = diagonale supérieure (Attention: c[:, -1] n'est pas utilisé)
                     d = second membre
    Returns:
        x : Tableau (M, N) solution
    """
    M, N = d.shape
    
    ac = a.copy()
    bc = b.copy()
    cc = c.copy()
    dc = d.copy()
    
    # 1. Descente (Forward Elimination) - Vectorisée sur M
    for i in range(1, N):
        m = ac[:, i] / bc[:, i-1]
        
        bc[:, i] = bc[:, i] - m * cc[:, i-1]
        dc[:, i] = dc[:, i] - m * dc[:, i-1]
        
    # 2. Remontée (Back Substitution)
    x = np.zeros((M, N))
    x[:, -1] = dc[:, -1] / bc[:, -1]
    
    for i in range(N-2, -1, -1):
        x[:, i] = (dc[:, i] - cc[:, i] * x[:, i+1]) / bc[:, i]
        
    return x