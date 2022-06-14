import numpy as np
import matplotlib.pyplot as plt 

def initialize(dx, dy, D, C, sources, gridmap):

    dx2, dy2 = dx * dx, dy * dy
    dt = dx2 * dy2 / (2 * D * (dx2 + dy2)) 
    
    # initialize u
    phi = np.zeros(gridmap.shape)
    for i, concentration in enumerate(C):
        phi[sources[i][0], sources[i][1]] = concentration

    return phi, dt, dx2, dy2  

def update(phi, gridmap, D, dt, dx2, dy2):
    phi = phi.copy()
    phi_sum = phi.sum()
    phi[np.where(gridmap == 1)] = 0

    # Propagate with forward-difference in time, central-difference in space
    phi[1:-1, 1:-1] = phi[1:-1, 1:-1] + D * dt * (
        (phi[2:, 1:-1] - 2 * phi[1:-1, 1:-1] + phi[:-2, 1:-1]) / dx2 + 
        (phi[1:-1, 2:] - 2 * phi[1:-1, 1:-1] + phi[1:-1, :-2]) / dy2
    )

    # energy redistribution
    phi = phi / phi.sum() * phi_sum
    return abs(phi)

# how to visualize it as an animation and show both diffusion and maze on one image
def show(phi, dt, gridmap, t=0):
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    im = ax.imshow(phi, cmap=plt.get_cmap('hot'), interpolation='bicubic',  alpha=1)
    ax.set_axis_off()
    ax.set_title(f'{t * dt * 1000:.1f} ms | E {np.sum(phi):.2f}')
    plt.imshow(gridmap, cmap='Greys', alpha=0.5)
    plt.show()

def evolve(phi, gridmap, D, dt, dx2, dy2, n_steps=1, show_each=100, show_plot=False):
    if show_plot:
        show(phi, dt, gridmap)
    for t in range(n_steps):
        phi = update(phi, gridmap, D, dt, dx2, dy2)
        if show_plot and t % show_each == 0:
            show(phi, dt, gridmap, t)
    return phi