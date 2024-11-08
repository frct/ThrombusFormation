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

custom_colors = ['white', 'tan', 'orange']  # Colors corresponding to values 1 and 2
custom_cmap = ListedColormap(custom_colors)

max_dim = (7.5,8.75)

fig = plt.figure(figsize = max_dim, dpi=600)

ax1 = fig.add_axes((0.1,0.8,0.7,0.2))
ax_CB1 = fig.add_axes((0.82,0.75,0.02,0.1))
ax_CB2 = fig.add_axes((0.92,0.75,0.02,0.1))
ax2 = fig.add_axes((0.1,0.5,0.7,0.2))


font_small = 8
font_mid = 10
font_large = 12

save_format = 'svg'


F, ux, uy, vel,_ = pickle.load(open('Initial flow.pkl', 'rb'))

Ny, Nx = np.shape(ux)
x,y = np.meshgrid(np.arange(Nx), np.arange(Ny))

Δx = 1e-6
Cs = 1 / np.sqrt(3) # celerity of sound in lu/ts
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

reference_vel = np.round(np.max(velocity),3)

INJURY_LENGTH = 50

INJURY_START = (Nx - INJURY_LENGTH) // 2
INJURY_END = INJURY_START + INJURY_LENGTH

density = np.zeros((Ny,Nx))    
density[0,:] = 1 # vessel walls
density[-1,:] = 1
#density[0,INJURY_START:INJURY_END] = 0
density[1,INJURY_START:INJURY_END] = 0.3
activation = np.zeros_like(density)
activation[1,INJURY_START:INJURY_END] = 1

PLOS_dim = (7.5,8.75)


fig = plt.figure(figsize = PLOS_dim,dpi=300)

ax1 = fig.add_axes((0.1,0.8,0.7,0.2))
ax_CB1 = fig.add_axes((0.82,0.75,0.02,0.1))
ax_CB2 = fig.add_axes((0.92,0.75,0.02,0.1))
ax2 = fig.add_axes((0.1,0.5,0.7,0.2))
#ax1.set_aspect('equal')

vessel_walls = np.zeros((Ny,Nx))
vessel_walls[0,:] = 1
vessel_walls[-1,:] = 1
vessel_walls[0,103:153] = 2 # injury patch



vessel_walls_cm = ax1.imshow(vessel_walls[:,:], cmap=custom_cmap)
    
mask = (activation == 0) | (vessel_walls > 0)
mask_velocity = (activation>0) | (vessel_walls>0)
ylabels = [str(i) for i in range(0,65,16)]
ax1.set_yticks(np.arange(-0.5,65,16),labels=ylabels, fontsize=8)
xlabels = [str(i) for i in range(0,257,64)]
ax1.set_xticks(np.arange(-0.5,257,64),labels=xlabels, fontsize=8)
ax1.set_ylabel('[μm]', fontsize=10)
ax1.set_xlabel('[μm]', fontsize=10)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

    
Z = np.ma.masked_where(mask, activation)
cs = ax1.imshow(-1*Z, origin ='lower', cmap = plt.cm.cividis, vmin =-1, vmax = 0)
#cax_activation = fig.add_axes([ax1.get_position().x1+0.1,ax1.get_position().y0,0.02,ax.get_position().height])
cb_activation = plt.colorbar(cs, cax=ax_CB1)


velocity_masked = np.ma.masked_where(mask_velocity, np.sqrt(ux**2+uy**2))
velocity_cm = ax1.imshow(velocity_masked, origin = 'lower', cmap = plt.cm.Reds, vmin = 0, vmax = reference_vel)

cb_vel = plt.colorbar(velocity_cm, cax=ax_CB2)
x_stride = 20
y_stride = 3


vel_field = ax1.quiver(x[1:-1:y_stride, ::x_stride],y[1:-1:y_stride, ::x_stride],ux[1:-1:y_stride, ::x_stride],uy[1:-1:y_stride, ::x_stride], scale=reference_vel*15)
scale_bar = ax1.quiverkey(vel_field, 0.88, 0.95, reference_vel, f'{reference_vel} m/s', coordinates='figure', labelpos='E')
scale_bar = ax1.quiverkey(vel_field, 0.88, 0.9, reference_vel/2, f'{reference_vel/2} m/s', coordinates='figure', labelpos='E')
scale_bar = ax1.quiverkey(vel_field, 0.88, 0.85, reference_vel/4, f'{reference_vel/4} m/s', coordinates='figure', labelpos='E')

'''############################################################################

second velocity field

############################################################################'''

load_name = 'clot retraction model/Simulation of clot retraction flow depdendent with detachment model with threshold = 1.1.pkl'
simulation = pickle.load(open(load_name, 'rb'))

activation = simulation['activation_frames'][:,:,-1]
ux = simulation['ux_frames'][:,:,-1] / Δt * Δx
uy = simulation['uy_frames'][:,:,-1] / Δt * Δx


vessel_walls_cm = ax2.imshow(vessel_walls[:,:], cmap=custom_cmap)
    
mask = (activation == 0) | (vessel_walls > 0)
mask_velocity = (activation>0) | (vessel_walls>0)
ylabels = [str(i) for i in range(0,65,16)]
ax2.set_yticks(np.arange(-0.5,65,16),labels=ylabels, fontsize=8)
xlabels = [str(i) for i in range(0,257,64)]
ax2.set_xticks(np.arange(-0.5,257,64),labels=xlabels, fontsize=8)
ax2.set_ylabel('[μm]', fontsize=10)
ax2.set_xlabel('[μm]', fontsize=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

    
Z = np.ma.masked_where(mask, activation)
cs = ax2.imshow(-1*Z, origin ='lower', cmap = plt.cm.cividis, vmin =-1, vmax = 0)


velocity_masked = np.ma.masked_where(mask_velocity, np.sqrt(ux**2+uy**2))
velocity_cm = ax2.imshow(velocity_masked, origin = 'lower', cmap = plt.cm.Reds, vmin = 0, vmax = reference_vel)


x_stride = 20
y_stride = 3


vel_field = ax2.quiver(x[1:-1:y_stride, ::x_stride],y[1:-1:y_stride, ::x_stride],ux[1:-1:y_stride, ::x_stride],uy[1:-1:y_stride, ::x_stride], scale=reference_vel*15)


fig.savefig('Figure1A.svg')

# fig, ax = plt.subplots()
# vessel_walls = np.ma.masked_where((density<1), density)
# initial_plug = np.ma.masked_where((density==0)|(density==1), activation)
# velocity_cm = ax.imshow(velocity, origin = 'lower', cmap = plt.cm.Reds,vmin = 0, vmax = 0.016)
# cbar = plt.colorbar(velocity_cm, ax=ax, label='Velocity Magnitude (m/s)', shrink = 0.4)
# vessel_cm = ax.imshow(vessel_walls, origin='lower', cmap='viridis_r')
# thrombus_cm = ax.imshow(-1 * initial_plug, origin ='lower', cmap = plt.cm.cividis, vmin =-1, vmax = 0)
# cbar_ac = plt.colorbar(thrombus_cm, ax=ax, label='activation_state', shrink = 0.4)

# x_stride = 20
# y_stride = 3


# vel_field = ax.quiver(x[1:-1:y_stride, ::x_stride],y[1:-1:y_stride, ::x_stride],ux[1:-1:y_stride, ::x_stride],uy[1:-1:y_stride, ::x_stride], scale = reference_vel)
# scale_bar = ax.quiverkey(vel_field, 0.8, 1.5, reference_vel, f'{reference_vel} m/s', coordinates='axes', labelpos='E')

# density_frames, activation_frames, ux_frames, uy_frames, clot_size = pickle.load(open('simulation of flow dependent activation', 'rb'))

# ux = ux_frames[-1,:,:] / Δt * Δx
# uy = uy_frames[-1,:,:] / Δt * Δx

# density = density_frames[-1,:,:]
# activation = activation_frames[-1,:,:]

# velocity = np.sqrt(ux**2 + uy**2)

# fig, ax = plt.subplots()
# vessel_walls = np.ma.masked_where((density<1), density)
# initial_plug = np.ma.masked_where((density==0)|(density==1), activation)
# velocity_cm = ax.imshow(velocity, origin = 'lower', cmap = plt.cm.Reds,vmin = 0, vmax = 0.016)
# cbar = plt.colorbar(velocity_cm, ax=ax, label='Velocity Magnitude (m/s)', shrink = 0.4)
# vessel_cm = ax.imshow(vessel_walls, origin='lower', cmap='viridis_r')
# thrombus_cm = ax.imshow(-1*initial_plug, origin ='lower', cmap = plt.cm.cividis, vmin =-1, vmax = 0)
# cbar_ac = plt.colorbar(thrombus_cm, ax=ax, label='activation_state', shrink = 0.4)


# x_stride = 20
# y_stride = 3

# vel_field = ax.quiver(x[1:-1:y_stride, ::x_stride],y[1:-1:y_stride, ::x_stride],ux[1:-1:y_stride, ::x_stride],uy[1:-1:y_stride, ::x_stride]) #, scale=reference_vel)
# #scale_bar = ax.quiverkey(vel_field, 0.8, 1.5, reference_vel, f'{reference_vel} m/s', coordinates='axes', labelpos='E')


# ### calibration of binding and detachment rates

# # MAX_NUMBER_NEW_PLATELETS_PER_SECOND = 1000
# # Δt = 1 / MAX_NUMBER_NEW_PLATELETS_PER_SECOND # time steps in s

# # binding_time_sec = 1 # 50 ms
# # binding_time = binding_time_sec / Δt
# # lambda_val = 1 / binding_time


# # # Define the equation as a Python function
# # def equation_to_solve(β, lambda_val):
# #     density = 0.3
# #     activation = 1
# #     p_i = β * density * activation 
# #     return 1 - (1 - p_i) * (1 - p_i / np.sqrt(2))**2 - lambda_val

# # solution = fsolve(equation_to_solve, x0=0.0, args=(lambda_val,))

# # binding_times = [i * Δt for i in np.arange(8,80)]

# # betas = []

# # for T in binding_times:
# #     t = T / Δt
# #     β = fsolve(equation_to_solve, x0=0.0, args=(1/t,))
# #     betas.append(β)
    
# # plt.figure()
# # plt.plot(betas, binding_times, 'black')
# # plt.xlabel('Binding rate, β')
# # plt.ylabel('Expected binding time (s)')
# # plt.ylim([0,0.08])
# # plt.xlim([0,0.15])

# # '''############################################################################

# # PLOT of p(DET) = f(E(Δt_det))

# # ############################################################################'''

# # Δt = 0.01

# # savefile = 'pd=f(detachment_time).pkl'

# # try:
# #     [time, p] = pickle.load(open(savefile, 'rb'))
# # except:
# #     test_p = np.linspace(0.01,1,1000)**2

# #     Expected_wait = np.zeros(len(test_p))        
# #     for i,p in enumerate(test_p):
# #         Expected_wait[i] = GetExpectedDetachmentTime(p, Δt)
# #     time = Expected_wait[::-1]
# #     p = test_p[::-1]           
# #     pickle.dump([time,p], open(savefile, 'wb'))

# # plt.figure()
# # plt.plot(time, p, '-', color='black')
# # plt.xlabel('E($Δt_d$) [s]')
# # plt.ylabel('p')
# # plt.xlim([0.1,2])
# # plt.ylim([0,0.1])
# # plt.xticks([0.5,1,1.5,2])

# # p_flow = np.arange(0.0001, 0.005, 0.00001)
# # detachment_time = 1 / p_flow * Δt

# # plt.figure()
# # plt.plot(p_flow, detachment_time, 'black')
# # plt.ylabel('Expected detachment time (s)')
# # plt.xlabel('p_flow')
# # plt.ylim([0,10])
# # plt.xlim([0,0.005])