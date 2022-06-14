from curses import KEY_HELP
from diffusion import *
from cells import *
import time
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import ipywidgets as widgets
import os 


def create_gridmap(path):
    global gridmap
    gridmap = plt.imread(path)[:,:,0]
    gridmap = (np.invert(gridmap.astype(bool))).astype(np.uint8)
    plt.imshow(gridmap, cmap='gray')
    plt.show()

def show_sources(sources):
    plt.imshow(gridmap, cmap='gray')
    sources = np.array(sources)
    xs = sources[:,1]
    ys = sources[:,0]
    plt.scatter(xs, ys)
    plt.show()

def run_experiment(Diffusivity=(0.1,1000.0), Concentration=(1000,7000), n=(1,100), k_max = (0.0,1.0), V_max=(0.0,1.0), self_generated=False, n_runs = (500,10000), save=False, folder_name=" ", sources=None, kinetics_plot=False, unequal_sources=False):
    global phi

    # set the of parameters
    dx = dy = 0.1
    show_each = 100

    # initialize the concentrations as a list of concentrations for each source
    concentrations = [Concentration]
    k = len(sources)

    # if sources are unequal, initialize the concentrations as a list of concentrations for each source differently 
    if unequal_sources:
        for __ in range(1, k):
            concentrations.append(Concentration / 14)
    # if equal, set them same
    else:
        concentrations = concentrations * k

    # initialize the gridmap with sources
    phi, dt, dx2, dy2 = initialize(dx, dy, Diffusivity, concentrations, sources, gridmap)
    phi_mean = phi.mean()
    
    # initialize cells
    cells = []
    while len(cells) != n:
        xs = np.linspace(0, 25, 5, dtype=int)
        ys = np.linspace(30, 70, 20, dtype=int)
        x = np.random.choice(xs, size=1)[0]
        y = np.random.choice(ys, size=1)[0]
        if gridmap[y,x] == 0:
            cell = Cell(x, y, self_generated, k_max, V_max)
            cells.append(cell)
    
    sums = []

    # run experiment
    for t in range(n_runs):
        
        # update concentration map
        phi = update(phi, gridmap, Diffusivity, dt, dx2, dy2)
        sums.append(phi.sum())

        # move cells according to sense function
        xs, ys = [], []
        for cell in cells:
            cell.step(phi, gridmap)
            if t % show_each == 0:
                x = cell.x
                xs.append(x)
                y = cell.y
                ys.append(y)

        # visualize only certain time frames
        if t == 0 or t % show_each == 0:
            fig, ax = plt.subplots(figsize=(10,10))
            data = np.ma.masked_where(gridmap != 0, phi)
            im = ax.imshow(data, interpolation='bicubic', vmin=0, vmax=phi_mean)
            ax.set_title(f'Time: {t * dt * 1000:.1f} ms') 
            ax.scatter(xs, ys, c='tab:red')
            fig.colorbar(im)
            if save:
                os.makedirs(f'{folder_name}', exist_ok=True)
                plt.savefig(f'{folder_name}/img_{t}')
            plt.show()
            time.sleep(0.1)
            clear_output(wait=True)

    if kinetics_plot:
        # plot sums vs time
        plt.plot(sums)
        # set y range
        plt.ylim(0, max(sums) * 1.1)
        plt.title('Sum of concentration over time')
        plt.xlabel('Time [steps]')
        plt.ylabel('Sum of concentration')
        plt.savefig('sum_plot.png')
        plt.show()
        

def kinetics(k_max=(0,500), V_max=(0.1,1)):
    dx = dy = 0.1
    Diffusivity = 500
    concentrations = [1000]
    sources = sources = [[80, 370]]
    # initialize phi again 
    phi, dt, dx2, dy2 = initialize(dx, dy, Diffusivity, concentrations, sources, gridmap)
    # set the of parameters
    concentrations = sorted(set(phi.flatten()))
    rates = [(V_max * c / (c + k_max)) for c in concentrations]
    plt.plot([0, 370], [V_max / 2, V_max / 2],'k--', [k_max, k_max], [0, V_max / 2], 'k:')
    plt.legend(('$0.5 V_{m}$', '$k_{m}$'))
    plt.plot(concentrations, rates)
    plt.title('Michaelis-Menten kinetics')
    plt.xlabel('Concentration [units per mm]')
    plt.ylabel('Rate of enzymatic activity')
    plt.show()