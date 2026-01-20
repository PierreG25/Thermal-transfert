import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import matplotlib as mpl

from components.computation.solve_psi import *
from components.computation.solve_temperature import *
from components.computation.compute_velocity import *
from components.computation.compute_nusselt import *
from components.computation.solve_vorticity import *

def global_resolution(nx, ny, Lx, Ly, dt, U0, Ra):
    
    dx, dy = Lx/(nx-1), Ly/(ny-1)


    rho = 1 # Air
    mu = 1e-5 # Air
    Pr = 0.71
    Re = rho * U0 * (Lx + Ly) / (2 * mu)
    Ri = Ra / (Re**2 * Pr)


    alpha_sor = 1.74
    tol_sor = 1e-6
    tol_steady_state = 1e-4
    max_iter = 100000

    T = np.zeros((nx, ny))
    w = np.zeros((nx, ny))
    psi = np.zeros((nx, ny))
    u, v = get_velocity(psi, dx, dy, U0)
    T[:, 0] = 0.0  # Paroi gauche chaude


    print("Le nombre de Courant est égale à : " + str(U0*dt/dy))

    print("Re = " + str(Re))

    assert U0*dt/dy < 1, f"Condition de stabilité CFL violée : {U0*dt/dy} "

    img_dic = {'T': [T], 'w': [w], 'psi': [psi], 'u': [u], 'v': [v]}

    n = 0

    res_w = []
    res_psi = []
    res_T = []
    res_w = []
    res_T = []


    while n <= max_iter:
        T_new = solve_adi_T(T, u, v, 1/(Re*Pr), dt, dx, dy)
        
        w_new = solve_adi_w(w, T_new, psi, u, v, 1/Re, Ri, dt, dx, dy, U0)
        
        psi_new = solve_psi_SOR(psi, w_new, dx, dy, alpha_sor, tol_sor)
        
        u_new, v_new = get_velocity(psi_new, dx, dy, U0)

        res_w.append(np.linalg.norm(w_new - w)/np.linalg.norm(w))
        res_T.append(np.linalg.norm(T_new - T)/np.linalg.norm(T))
        
        if res_w[-1] < tol_steady_state and res_T[-1] < tol_steady_state:
                print(f"\nConvergence atteinte à l'itération {n} !")
                break


        if n % 10 == 0:
            img_dic['T'].append(T.copy())
            img_dic['w'].append(w.copy())
            img_dic['psi'].append(psi.copy())
            u_save, v_save = get_velocity(psi, dx, dy, U0)
            img_dic['u'].append(u_save.copy())
            img_dic['v'].append(v_save.copy())
            print(f"Itération {n}: Résidu w = {res_w[-1]:.2e}, Résidu T = {res_T[-1]:.2e}")
        
        T, w, psi, u, v = T_new, w_new, psi_new, u_new, v_new
        n += 1

    #plt.plot(res_T, label='Résidu Température')
    #plt.plot(res_w, label='Résidu Vorticité')
    #plt.yscale('log')
    #plt.xlabel('Itérations')
    #plt.ylabel('Résidu')
    #plt.legend()
    #plt.title('Convergence des résidus')
    #plt.grid()
    #plt.show()


    Nu = get_average_nusselt(T, dx)
    return Nu, Re, T.copy(), psi.copy()     

