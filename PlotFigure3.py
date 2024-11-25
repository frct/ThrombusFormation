# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 11:05:20 2024

@author: vf926215
"""


import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


max_dim = (7.5,8.75)

fig = plt.figure(figsize = max_dim,dpi=900)

ax1 = fig.add_axes((0.1,0.6,0.4,0.2))
ax2 = fig.add_axes((0.1,0.4,0.4,0.2))
ax3 = fig.add_axes((0.1,0.2,0.4,0.2))
ax4 = fig.add_axes((0.7,0.6,0.3,0.2))
ax5 = fig.add_axes((0.7,0.2,0.3,0.2))

font_tiny = 6
font_small = 8
font_mid = 10
font_large = 12

save_format = 'svg'

'''###########################################################
FIGURE 3A
###########################################################'''

Nx = 256
Ny = 64

X,Y = np.meshgrid(np.arange(Nx), np.arange(Ny))

heights = [0, 15, 30]

Δx_USI = 1e-6 # lattice unit (lu) size in m
HEIGHT_USI = (Ny-2) * Δx_USI
RADIUS_USI = HEIGHT_USI / 2 # m
γ_USI = 1000 # s-1
U_MAX_USI = γ_USI * RADIUS_USI / 2 # m/s
INJURY_LENGTH = 50
INJURY_START = (Nx - INJURY_LENGTH) // 2
INJURY_END = INJURY_START + INJURY_LENGTH



_, _, _, _, Δt_LBM = pickle.load(open('Stenosis/Poiseuille flow.pkl', 'rb'))

# initialise velocity

ux = np.zeros((Ny,Nx))
uy = np.zeros((Ny,Nx))

for idy in range(1,Ny-1):
    y = (idy-1) * Δx_USI + Δx_USI / 2
    r = y - RADIUS_USI
    ux[idy,:] = U_MAX_USI * (1-r**2/RADIUS_USI**2)


velocity_USI = np.sqrt(ux**2+uy**2)




vessel_walls = np.zeros((Ny,Nx))
vessel_walls[0,:] = 1
vessel_walls[-1,:] = 1

custom_colors = ['white', 'tan']  # Colors corresponding to values 0 and 1
custom_cmap = ListedColormap(custom_colors)


x_stride = 20
y_stride = 3
reference_vel = 0.016

for i, (h,ax) in enumerate(zip(heights,[ax1,ax2,ax3])):
    if h > 0:
        vessel_walls[:h,INJURY_START:INJURY_END] = 1
        vessel_walls_cm = ax.imshow(vessel_walls, cmap=custom_cmap, origin = 'lower')
        F, ux, uy, velocity = pickle.load(open(f'Stenosis/Flow for height={h}.pkl', 'rb'))
        ux *= Δx_USI / Δt_LBM 
        uy *= Δx_USI / Δt_LBM 
        velocity_USI = velocity * Δx_USI / Δt_LBM 

        
    vessel_walls_cm = ax.imshow(vessel_walls, cmap=custom_cmap, origin='lower')
    velocity = np.ma.masked_where(vessel_walls, velocity_USI*1000)
    velocity_cm = ax.imshow(velocity, origin = 'lower', cmap = plt.cm.Reds, vmin=0, vmax = 20)
    
    vel_field = ax.quiver(X[1:-1:y_stride, ::x_stride],Y[1:-1:y_stride, ::x_stride],ux[1:-1:y_stride, ::x_stride],uy[1:-1:y_stride, ::x_stride], scale=reference_vel*15)
    scale_bar = ax.quiverkey(vel_field, 0.8, 1.1, reference_vel, f'{reference_vel} m/s', coordinates='axes', labelpos='E')
    
    ylabels = [str(i) for i in range(0,65,16)]
    ax.set_yticks(np.arange(-0.5,65,16),labels=ylabels, fontsize=8)
    xlabels = [str(i) for i in range(0,257,64)]
    ax.set_xticks(np.arange(-0.5,257,64),labels=xlabels, fontsize=8)
    ax.set_ylabel('[μm]', fontsize=10)
    ax.set_xlabel('[μm]', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

'''#####################################################################
FIGURES 3B AND C
#####################################################################'''

heights =  [5, 10, 15, 20, 25, 30, 35, 40]
v_peak = np.zeros_like(heights, dtype=float)
v_inlet = np.zeros_like(heights, dtype=float)


for i,h in enumerate(heights):
    F, ux, uy, velocity = pickle.load(open(f'Stenosis/Flow for height={h}.pkl', 'rb'))
    v_peak[i] = np.max(velocity) * Δx_USI / Δt_LBM * 1000 
    v_inlet[i] = np.mean(velocity[:,0]) * Δx_USI / Δt_LBM * 1000

ax4.plot(heights, v_peak, '-ok')
ax5.plot(heights, v_inlet, '-ok')

for ax in [ax4,ax5]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('$H_{obst} [\mu m]$', fontsize = font_mid)
    ax.set_xticks(heights)
    ax.set_xticklabels(heights, fontsize = font_small)
    
ax4.set_ylim([0,20])
ax4.set_yticklabels([0,5,10,15,20], fontsize=font_small)
ax4.set_ylabel('$v_{peak}$ [mm/s]', fontsize = font_mid)

ax5.set_ylim([0,12])
ax4.set_yticklabels([0,2,4,6,8,10,12], fontsize=font_small)
ax5.set_ylabel('$v_{inlet}$ [mm/s]', fontsize= font_mid)

'''############################################################################
ADD TITLES AND SAVE
############################################################################'''

ax1.set_title('A', fontsize=font_large, loc='left', y=1, fontweight='bold')
ax4.set_title('B', fontsize=font_large, loc='left', y=1, fontweight='bold')
ax5.set_title('C', fontsize=font_large, loc='left', y=1, fontweight='bold')

fig.savefig(f'Figure3.{save_format}', format = save_format)