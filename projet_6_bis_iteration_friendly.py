import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import matplotlib as mpl

from components.computation.solve_psi import *
from components.computation.solve_temperature import *
from components.computation.compute_velocity import *
from components.computation.solve_vorticity import *
from components.computation.compute_nusselt import *

def global_resolution(nx, ny, Lx, Ly, dt, nu, Re, Ra):
    
    dx, dy = Lx/(nx-1), Ly/(ny-1)

    Pr = 0.71
    U0 = Re * nu /(Lx+Ly)*2
    Ri = Ra / (Re**2 * Pr)
    alpha_sor = 1.74
    tol_sor = 1e-6
    tol_steady_state = 1e-4
    tol_Nu = 1e-3
    max_iter = 100000

    T = np.zeros((nx, ny))
    w = np.zeros((nx, ny))
    psi = np.zeros((nx, ny))
    u, v = get_velocity(psi, dx, dy, U0)
    T[:, 0] = 1.0  # Paroi gauche chaude

    img_dic = {'T': [T], 'w': [w], 'psi': [psi], 'u': [u], 'v': [v]}

    res_w = []
    res_T = []
    res_w = []
    res_T = []
    res_Nu = []
    n = 0
    while n <= max_iter:
        C = max(np.max(np.abs(u)), np.max(np.abs(v))) *dt/dx
        if C>1:
            print(f" Re = {Re:.0f}, Ra = {Ra:.2e} : Condition CFL brisée à l'itération {n} : C = {C:.3f}")
            return

        T_new = solve_adi_T(T, u, v, 1/(Re*Pr), dt, dx, dy)
        
        w_new = solve_adi_w(w, T_new, psi, u, v, 1/Re, Ri, dt, dx, dy, U0)
        
        psi_new = solve_psi_SOR(psi, w_new, dx, dy, alpha_sor, tol_sor)
        
        u_new, v_new = get_velocity(psi_new, dx, dy, U0)

        res_w.append(np.linalg.norm(w_new - w)/np.linalg.norm(w))
        res_T.append(np.linalg.norm(T_new - T)/np.linalg.norm(T))

        Nu_h, Nu_c = get_average_nusselt(T_new, dx)
        res_Nu.append(abs(Nu_h-Nu_c)/Nu_h)
        
        if res_w[-1] < tol_steady_state and res_T[-1] < tol_steady_state and res_Nu[-1]<tol_Nu:
                print(f"\nConvergence atteinte à l'itération {n} !")
                break

        if n % 10 == 0:
            img_dic['T'].append(T.copy())
            img_dic['w'].append(w.copy())
            img_dic['psi'].append(psi.copy())
            u_save, v_save = get_velocity(psi, dx, dy, U0)
            img_dic['u'].append(u_save.copy())
            img_dic['v'].append(v_save.copy())
            print(f"Itération {n}: Résidu Nu = {res_Nu[-1]:.2e}, Résidu w = {res_w[-1]:.2e}, Résidu T = {res_T[-1]:.2e}")
        
        T, w, psi, u, v = T_new, w_new, psi_new, u_new, v_new
        n += 1

    img_dic['res_T'] = res_T
    img_dic['res_w'] = res_w
    img_dic['res_Nu'] = res_Nu
    return U0, img_dic      
