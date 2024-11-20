# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:42:31 2024

Plot figure 1

@author: vf926215
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import ListedColormap
from RunSimulation import GetBeta, GetPDetach

custom_colors = ['white', 'tan', 'orange']  # Colors corresponding to empty, vessel walls, and platelet
custom_cmap = ListedColormap(custom_colors)

max_dim = (7.5,8.75)

fig = plt.figure(figsize = max_dim, dpi=900)

# ax1 = fig.add_axes((0.1,0.8,0.7,0.2))
# ax2 = fig.add_axes((0.1,0.5,0.7,0.2))
ax1 = fig.add_axes((0.2,0.8,0.4,0.1))
ax2 = fig.add_axes((0.2,0.6,0.4,0.1))
ax_CB1 = fig.add_axes((0.65,0.6,0.02,0.25))
ax_CB2 = fig.add_axes((0.02,0.6,0.02,0.25))
ax3 = fig.add_axes((0.1,0.1,0.3,0.2))
ax4 = fig.add_axes((0.6,0.1,0.3,0.2))

font_tiny = 6
font_small = 8
font_mid = 10
font_large = 12

save_format = 'svg'


'''############################################################################
FIGURE 1A TOP
############################################################################'''

# "Initial flow.pkl" is generated when RunSimulation is called

F, ux, uy, _, _ = pickle.load(open('Initial flow.pkl', 'rb'))
    
Ny, Nx = np.shape(ux)

Δx = 1e-6 # grid spacing of 1 μm
Cs = 1 / np.sqrt(3) # celerity of sound in lattice units per timestep
ρ_USI = 1060 # kg/m3
μ_USI = 4e-3 # extrapolation from Cherry 2013 and lab calculation for in vitro experiments
NU_USI = μ_USI / ρ_USI
τ = 0.809
nu = Cs**2 * (τ-1/2)
C_nu = NU_USI / nu
Δt = Δx**2 / C_nu
    
# convert ux and uy from LBM units into m/s

ux = ux / Δt * Δx
uy = uy / Δt * Δx
velocity = np.sqrt(ux**2 + uy**2)

# reference value shared between Fig1A top and bottom
reference_vel = np.round(np.max(velocity),3) 

INJURY_LENGTH = 50

INJURY_START = (Nx - INJURY_LENGTH) // 2
INJURY_END = INJURY_START + INJURY_LENGTH

density = np.zeros((Ny,Nx))    
density[0,:] = 1 # vessel walls
density[-1,:] = 1
density[1,INJURY_START:INJURY_END] = 0.3
activation = np.zeros_like(density)
activation[1,INJURY_START:INJURY_END] = 1


vessel_walls = np.zeros((Ny,Nx))
vessel_walls[[0,-1],:] = 1
vessel_walls[0,INJURY_START:INJURY_END] = 2 # injured patch
vessel_walls_cm = ax1.imshow(vessel_walls[:,:], cmap=custom_cmap)
    
# HEATMAPS OF ACTIVATION AND VELOCITY MAGNITUDE
    
activation_masked = -1 * np.ma.masked_where(activation == 0, activation) # flipped so that the colormap goes from yellow blue
activation_colormap = ax1.imshow(activation_masked, origin ='lower', cmap = plt.cm.cividis, vmin =-1, vmax = 0)
colorbar_activation = plt.colorbar(activation_colormap, cax=ax_CB1)
colorbar_activation.set_ticks([0,-0.2,-0.4,-0.6,-0.8,-1])
colorbar_activation.ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1], fontsize=font_small)

velocity_masked = np.ma.masked_where((activation>0) | (vessel_walls>0), velocity)
velocity_colormap = ax1.imshow(velocity_masked, origin = 'lower', cmap = plt.cm.Reds, vmin = 0, vmax = reference_vel)
colorbar_velocity = plt.colorbar(velocity_colormap, cax=ax_CB2)
colorbar_velocity.mappable.set_clim(vmin=0, vmax=reference_vel)  # Adjust the range of the color scale
colorbar_velocity.update_normal(velocity_colormap)  # Update the colorbar to reflect changes
colorbar_velocity.set_ticks([0, 0.004, 0.008, 0.012, 0.016])  # Specify custom tick positions
colorbar_velocity.ax.set_yticklabels([0, 4, 8, 12, 16], fontsize=font_small)  # Custom tick labels and fontsize



# QUIVER PLOT OF VELOCITY

x,y = np.meshgrid(np.arange(Nx), np.arange(Ny))

# common to top and bottom
x_stride = 20
y_stride = 3
quiver_scale = 15 

vel_field = ax1.quiver(x[1:-1:y_stride, ::x_stride],
                        y[1:-1:y_stride, ::x_stride],
                        ux[1:-1:y_stride, ::x_stride],
                        uy[1:-1:y_stride, ::x_stride],
                        scale=reference_vel*quiver_scale)

scale_bar = ax1.quiverkey(vel_field, 0.8, 0.9, reference_vel, f'{reference_vel} m/s', coordinates='figure', labelpos='E')
scale_bar = ax1.quiverkey(vel_field, 0.8, 0.85, reference_vel*3/4, f'{reference_vel} m/s', coordinates='figure', labelpos='E')
scale_bar = ax1.quiverkey(vel_field, 0.8, 0.8, reference_vel/2, f'{reference_vel/2} m/s', coordinates='figure', labelpos='E')
scale_bar = ax1.quiverkey(vel_field, 0.8, 0.75, reference_vel/4, f'{reference_vel/4} m/s', coordinates='figure', labelpos='E')


'''############################################################################
FIGURE 1A BOTTOM
############################################################################'''

# Load a previously saved simulation of the flow dependent model and extract
# the last frames of activation and velocity

simulation = pickle.load(open('Fig1ASimulation.pkl', 'rb'))
activation = simulation['activation_frames'][:,:,-1]
ux = simulation['ux_frames'][:,:,-1] / Δt * Δx
uy = simulation['uy_frames'][:,:,-1] / Δt * Δx

vessel_walls_cm = ax2.imshow(vessel_walls[:,:], cmap=custom_cmap)
    
mask_velocity = (activation>0) | (vessel_walls>0)
activation_masked = -1 * np.ma.masked_where(activation == 0, activation)
ax2.imshow(activation_masked, origin ='lower', cmap = plt.cm.cividis, vmin =-1, vmax = 0)

velocity_masked = np.ma.masked_where(mask_velocity, np.sqrt(ux**2+uy**2))
ax2.imshow(velocity_masked, origin = 'lower', cmap = plt.cm.Reds, vmin = 0, vmax = reference_vel)


ax2.quiver(x[1:-1:y_stride, ::x_stride],
            y[1:-1:y_stride, ::x_stride],
            ux[1:-1:y_stride, ::x_stride],
            uy[1:-1:y_stride, ::x_stride],
            scale=reference_vel*quiver_scale)


'''############################################################################
APPLY SAME FORMAT TO TOP AND BOTTOM SUB-FIGURES
############################################################################'''
for ax in [ax1, ax2]:

    xlabels = [str(i) for i in range(0,257,64)]
    ax.set_xticks(np.arange(-0.5,257,64),labels=xlabels, fontsize=font_tiny)
    
    ylabels = [str(i) for i in range(0,65,16)]
    ax.set_yticks(np.arange(-0.5,65,16),labels=ylabels, fontsize=font_tiny)
    
    ax.tick_params(axis='x', width=0.5, length = 1)
    ax.tick_params(axis='y', width=0.5, length = 1) 
    
    ax.set_ylabel('[μm]', fontsize = font_mid)
    ax.set_xlabel('[μm]', fontsize = font_mid)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


'''############################################################################
FIGURE 1B IS A DIAGRAM DRAWN IN INKSCAPE
###########################################################################'''




'''############################################################################
FIGURE 1C: BINDING RATE = F(EXPECTED BINDING TIME)
############################################################################'''

Δt = 0.001
binding_times = [i * Δt for i in np.arange(15,80)]

betas = []

for t in binding_times:
    betas.append(GetBeta(t))
    
ax3.plot(binding_times, betas, 'black')
ax3.set_xlabel('expected binding time [ms]', fontsize = font_mid)
ax3.set_ylabel('binding rate, β', fontsize = font_mid)
ax3.set_xlim([0,0.08])
ax3.set_ylim([0,0.35])
ax3.set_xticks([0,0.02,0.04,0.06,0.08])
ax3.set_xticklabels([0,20,40,60,80],  fontsize=font_small)
ax3.set_yticks([0,0.1,0.2,0.3])
ax3.set_yticklabels([0,0.1,0.2,0.3], fontsize = font_small)


'''############################################################################
FIGURE 1D: DETACHMENT RATE = F(EXPECTED DETACHMENT TIME)
############################################################################'''

Δt = 0.01
detach_times = [i * Δt for i in np.arange(10,200)]

p_shear = []

for t in detach_times:
    p_shear.append(GetPDetach(t))


ax4.plot(detach_times, p_shear, color='black')
ax4.set_xlabel('expected detachment time [s]', fontsize = font_mid)
ax4.set_ylabel('detachment rate, $p_{shear}$', fontsize = font_mid)
ax4.set_xlim([0,2])
ax4.set_ylim([0,0.1])
ax4.set_xticks([0,0.5,1,1.5,2])
ax4.set_xticklabels([0,0.5,1,1.5,2], fontsize=font_small)
ax4.set_yticks([0, 0.02, 0.04, 0.06, 0.08, 0.1])
ax4.set_yticklabels([0, 0.02, 0.04, 0.06, 0.08, 0.1], fontsize=font_small)

'''############################################################################
ADD TITLES AND SAVE
############################################################################'''

ax1.set_title('A', fontsize=font_large, loc='left', y=1, fontweight='bold')
ax2.set_title('B', fontsize=font_large, loc='left', y=1, fontweight='bold')
ax3.set_title('C', fontsize=font_large, loc='left', y=1, fontweight='bold')
ax4.set_title('D', fontsize=font_large, loc='left', y=1, fontweight='bold')

fig.savefig(f'Figure1.{save_format}', format = save_format)