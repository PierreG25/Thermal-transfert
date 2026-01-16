import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animer_matrices(liste_matrices, dt, fps=20, save_path=None, Name=''):
    fig, ax = plt.subplots()
    
    # Initialisation de l'affichage avec la première matrice
    # cmap='viridis' est souvent mieux pour les données scientifiques que 'gray'
    im = ax.imshow(liste_matrices[0], animated=True, cmap='viridis', origin='lower', vmin = np.min(liste_matrices), vmax = np.max(liste_matrices))
    plt.colorbar(im) # Optionnel : affiche l'échelle des valeurs
    plt.title('Animation des Matrices')
    ax.axis('off')
    

    # Fonction de mise à jour appelée pour chaque frame
    def update(i):
        im.set_array(liste_matrices[i])
        ax.set_title(f"{Name}, t={i*dt*10:.1f}s")
        return [im]

    # Création de l'animation
    # interval = délai entre images en millisecondes (1000 / fps)
    ani = animation.FuncAnimation(fig, update, frames=len(liste_matrices), 
                        interval=1000/fps, blit=True)

    ani.save(save_path, writer='pillow', fps=fps)
    
    plt.show()
    return ani