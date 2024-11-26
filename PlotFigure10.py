# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 14:37:38 2024

@author: vf926215
"""


from RunSimulationV2 import RunSimulation
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl

mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

x_stride = 20
y_stride = 3
reference_vel = 0.016

BINDING_TIME_SEC = 0.02
DETACHMENT_TIME_SEC = 5
HALF_ACTIVATION_SEC = 0.2

T = 600

sliding_window_sec = 30
start_row = 1

averages = []
errors = []



vessel_walls = np.zeros((64,256))
vessel_walls[0,:] = 1
vessel_walls[-1,:] = 1

custom_colors = ['white', 'tan']  # Colors corresponding to values 0 and 1
custom_cmap = ListedColormap(custom_colors)
_, _, _, _, Δt_LBM = pickle.load(open('Poiseuille flow.pkl', 'rb'))
format_ = 'svg'

PLOS_dim = (7.5,8.75)
fig1 = plt.figure(figsize=PLOS_dim, dpi=300)
ax1 = fig1.add_axes((0.1,0.9,0.3,0.2))
ax2 = fig1.add_axes((0.1,0.7,0.3,0.2))
ax3 = fig1.add_axes((0.1,0.5,0.3,0.2))
ax4 = fig1.add_axes((0.1,0.3,0.3,0.2))

ax = [ax1, ax2, ax3, ax4]
ax1.set_title('A', fontsize=12, fontweight='bold')
ax2.set_title('B', fontsize=12, fontweight='bold')
ax3.set_title('C', fontsize=12, fontweight='bold')
ax4.set_title('D', fontsize=12, fontweight='bold')

save_name = f'clot retraction model/Simulation of clot retraction flow depdendent with detachment model with threshold = 1.1.pkl'

results = pickle.load(open(save_name, 'rb'))
Δt = results['Δt frame']

frames = [7, 15, 30,60]

for i,f in enumerate(frames):
    
    ax[i].set_aspect('equal')    
    vessel_walls_cm = ax[i].imshow(vessel_walls, cmap=custom_cmap)
    
    velocity = np.ma.masked_where(results['density_frames'][:,:,f] != 0,
                                  np.sqrt(
                                      results['ux_frames'][:,:,f]**2
                                      +
                                      results['uy_frames'][:,:,f]**2
                                      )
                                  / Δt_LBM * 1e-3)
    velocity_cm = ax[i].imshow(velocity, origin = 'lower', cmap = plt.cm.Reds, vmin=0, vmax = 16)
    
    ux = results['ux_frames'][:,:,f] / Δt_LBM * 1e-6
    uy = results['uy_frames'][:,:,f] / Δt_LBM * 1e-6
    Ny, Nx = np.shape(ux)
    x,y = np.meshgrid(np.arange(Nx), np.arange(Ny))
    vel_field = ax[i].quiver(x[1:-1:y_stride, ::x_stride],y[1:-1:y_stride, ::x_stride],ux[1:-1:y_stride, ::x_stride],uy[1:-1:y_stride, ::x_stride], scale=reference_vel*15)

    
    activation_masked = np.ma.masked_where((results['density_frames'][:,:,f] == 0), 
                                            results['activation_frames'][:,:,f])
    ax[i].imshow(activation_masked, origin='lower', cmap=plt.cm.cividis, vmax=1, vmin=0)
    
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['bottom'].set_visible(False)
    ax[i].spines['left'].set_visible(False)
    ylabels = [str(i) for i in range(0,65,16)]
    ax[i].set_yticks(np.arange(-0.5,65,16),labels=ylabels, fontsize=8)
    xlabels = [str(i) for i in range(0,257,64)]
    ax[i].set_xticks(np.arange(-0.5,257,64),labels=xlabels, fontsize=8)
    ax[i].set_ylabel('[μm]', fontsize=10)
    ax[i].set_xlabel('[μm]', fontsize=10)
    ax[i].set_title(f't = {int(f * Δt)} s')
    
    
    


scale_bar = ax2.quiverkey(vel_field, 0.88, 1.1, reference_vel, f'{reference_vel} m/s', coordinates='axes', labelpos='E')
# ax4.legend(fontsize=8)    
fig1.savefig(f'Figure10.{format_}', format=format_)    
