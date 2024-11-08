# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:43:28 2024

Plot Figure 2

@author: vf926215
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from RunSimulation import RunSimulation

# Get the default color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

max_dim = (7.5,8.75)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=max_dim, dpi=600)  # 10 cm wide and aspect ratio 2:3

font_small = 8
font_mid = 10
font_large = 12

save_format = 'svg'

''' ###########################################################################
FIGURE 2A

Simulate or load simulations of the constant binding + constant detachment
model for different detachment rates and compute + plot average growth

############################################################################'''

N_SIMULATIONS = 10
T_SEC = 60
BINDING_TIME_SEC = 0.05 
EXP_DET_TIMES_SEC = [1, 0.1, 0.05]


for i, t_det in enumerate(EXP_DET_TIMES_SEC):

    clot_sizes = []
    
    for simulation_nb in range(1,N_SIMULATIONS+1):
        save_name = f'CACD/Simulation {simulation_nb} with td = {t_det}.pkl'
        try:
            results = pickle.load(open(save_name, 'rb'))
        except: 
            results = RunSimulation(save_file=save_name,
                                    T = T_SEC,
                                    constant_binding = True, 
                                    constant_detachment = True,
                                    BINDING_TIME_SEC = BINDING_TIME_SEC,
                                    DETACHMENT_TIME_SEC = t_det,
                                    )
            
        clot_sizes.append(results['clot size'])
       
    times = [results['Δt'] * i for i in range(len(results['clot size']))]
    
    
    average_size = np.mean(np.array(clot_sizes), axis=0)
    stdev_size = np.std(np.array(clot_sizes), axis=0)
    ax1.plot(times, average_size, label = f'{t_det}s')
    ax1.fill_between(times, average_size - stdev_size, average_size + stdev_size, alpha=0.6)

        
ax1.set_xlabel('Time [s]', fontsize = font_mid)
ax1.set_ylabel('Thrombus size [μm$^2$]', fontsize = font_mid)
ax1.set_title('A', fontsize=font_large, loc='left', y=1, fontweight='bold')
ax1.legend(fontsize = font_small)
ax1.set_box_aspect(1)
ax1.set_xlim([0,60])
ax1.set_ylim([0,3000])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

'''############################################################################
FIGURE 2B

Simulate or load simulations of the activation-dependent binding + detachment
with a constant final level of platelet activation model for different 
detachment rates and compute + plot average growth

############################################################################'''

N_SIMULATIONS = 10
T_SEC = 60
BINDING_TIME_SEC = 0.05 
EXP_DET_TIMES_SEC = [0.05, 0.03, 0.02]
HALF_ACTIVATION_SEC = 0.2

for i, t_det in enumerate(EXP_DET_TIMES_SEC):
    
    clot_sizes = []

    
    for simulation_nb in range(1,N_SIMULATIONS+1):
        save_name = f'FA/test detachment rate/Simulation {simulation_nb} with E(Δt_det)={t_det}.pkl'
        
        try:
            results = pickle.load(open(save_name, 'rb'))
        except:               
            results = RunSimulation(save_file=save_name,
                                    T = T_SEC,
                                    constant_binding = False, 
                                    constant_detachment=False,
                                    final_activation = 1.0,
                                    BINDING_TIME_SEC = BINDING_TIME_SEC,
                                    DETACHMENT_TIME_SEC = t_det,
                                    HALF_ACTIVATION_SEC = HALF_ACTIVATION_SEC)
            
        clot_sizes.append(results['clot size'])
        
    
    times = [results['Δt'] * i for i in range(len(results['clot size']))]
    
    
    average_size = np.mean(np.array(clot_sizes), axis=0)
    stdev_size = np.std(np.array(clot_sizes), axis=0)
    ax2.plot(times, average_size, label = f'{t_det}s')
    ax2.fill_between(times, average_size - stdev_size, average_size + stdev_size, alpha=0.6)

        
ax2.set_xlabel('Time [s]', fontsize = font_mid)
ax2.set_title('B', fontsize=font_large, loc='left', y=1, fontweight='bold')
ax2.legend(fontsize = font_small)
ax2.set_box_aspect(1)
ax2.set_xlim([0,60])
ax2.set_ylim([-25,2500])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)


'''############################################################################
FIGURE 2C

Simulate or load simulations of the activation-dependent binding + detachment
with a constant final level of platelet activation model for different 
activation rates and compute + plot average growth

############################################################################'''

N_SIMULATIONS = 10
T_SEC = 60
BINDING_TIME_SEC = 0.05 
DETACHMENT_TIME_SEC = 1
HALF_ACTIVATION_TIMES_SEC = [1, 5, 10]

for i, t50 in enumerate(HALF_ACTIVATION_TIMES_SEC):
    
    clot_sizes = []

    
    for simulation_nb in range(1,N_SIMULATIONS+1):
        save_name = f'FA/test t50/Simulation {simulation_nb} with t50 = {t50}.pkl'
        
        try:
            results = pickle.load(open(save_name, 'rb'))
        except:               
            results = RunSimulation(save_file=save_name,
                                    T = T_SEC,
                                    constant_binding = False, 
                                    constant_detachment=False,
                                    final_activation = 1.0,
                                    BINDING_TIME_SEC = BINDING_TIME_SEC,
                                    DETACHMENT_TIME_SEC = DETACHMENT_TIME_SEC,
                                    HALF_ACTIVATION_SEC = t50)
            
        clot_sizes.append(results['clot size'])
        
    
    times = [results['Δt'] * i for i in range(len(results['clot size']))]
    
    
    average_size = np.mean(np.array(clot_sizes), axis=0)
    stdev_size = np.std(np.array(clot_sizes), axis=0)
    ax3.plot(times, average_size, label = f'{t50}s')
    ax3.fill_between(times, average_size - stdev_size, average_size + stdev_size, alpha=0.6)

        
ax3.set_xlabel('Time [s]', fontsize = font_mid)
ax3.set_title('C', fontsize=font_large, loc='left', y=1, fontweight='bold')
ax3.legend(fontsize = font_small)
ax3.set_box_aspect(1)
ax3.set_xlim([0,60])
ax3.set_ylim([0,2000])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)


fig.savefig(f'Figure2.{save_format}', format = save_format)