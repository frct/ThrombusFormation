# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:33:08 2024

@author: vf926215
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from RunSimulation import RunSimulation

max_dim = (7.5,4)

fig = plt.figure(figsize = max_dim, dpi=900)

ax1 = fig.add_axes((0.1,0.6,0.2,0.4))
ax2 = fig.add_axes((0.6,0.6,0.2,0.4))
ax3 = fig.add_axes((0.15,0.1,0.7,0.4))
ax3_CB = fig.add_axes((0.9,0.02,0.01,0.45))

# Get the default color cycle
colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

font_tiny = 6
font_small = 8
font_mid = 10
font_large = 12

save_format = 'svg'

'''##############################################################
FIGURE 5A: EITHER LOAD SAVED SIMULATIONS OR LAUNCH NEW ONES
##############################################################'''

BINDING_TIME_SEC = 0.05 
t_D = [1,0.1,0.03]
HALF_ACTIVATION_SEC = 0.2

n_repeats = 10
T = 600

sliding_window_sec = 1
start_row = 1

averages = []
errors = []

for i, DETACHMENT_TIME_SEC in enumerate(t_D):
    print(f'E(Δt_b)={DETACHMENT_TIME_SEC}')
    clot_growths = []
    detachments = []
    attachments = []
    net_growth = []
    density_ratio = []
    
    for rep in range(1,n_repeats+1):
        save_name = f'CAED/test detachment rate/Simulation {rep} of full model with td = {DETACHMENT_TIME_SEC}.pkl'
        try:
            results = pickle.load(open(save_name, 'rb'))
        except:               
            results = RunSimulation(save_file=save_name,
                                    T = T,
                                    want_flow=True,
                                    activation_dependent_binding=False,
                                    BINDING_TIME_SEC=BINDING_TIME_SEC,
                                    DETACHMENT_TIME_SEC = DETACHMENT_TIME_SEC,
                                    HALF_ACTIVATION_SEC = HALF_ACTIVATION_SEC,
                                    want_frames=True)
            
        clot_growths.append(results['clot size'])
        Nt = len(results['clot size'])
        Nt_window = int(sliding_window_sec / results['Δt'])
        Δclot = results['clot size'][1:] - results['clot size'][:-1]
        
        growth_rate = [np.sum(Δclot[t:t+Nt_window]) for t in range(Nt-Nt_window)]
        detachment_rate = [np.sum(results['detachment_events'][t:t+Nt_window])/sliding_window_sec for t in range(Nt-Nt_window)]
        binding_rate = [np.sum(results['binding_events'][t:t+Nt_window])/sliding_window_sec for t in range(Nt-Nt_window)]
        
        detachments.append(detachment_rate)
        attachments.append(binding_rate)
        net_growth.append(growth_rate)
        
        final_density = results['final density']
        
        nonzero_rows = np.nonzero(final_density[:-1,:])[0]
        # Get the maximum row index containing a non-zero value
        end_row = np.max(nonzero_rows)
        
        nonzero_cols = np.nonzero(final_density[1:-1,:])[1]
        start_col = np.min(nonzero_cols)
        end_col = np.max(nonzero_cols)
        area = (end_row - start_row + 1) * (end_col - start_col + 1)
        final_density = (final_density[start_row:end_row+1,start_col:end_col+1]>0)
        density_ratio.append(np.sum(final_density) / area)
        
    times = [results['Δt'] * i for i in range(len(results['clot size']))]
    
    
    avg = np.mean(np.array(clot_growths), axis=0)
    stdev = np.std(np.array(clot_growths), axis=0)
    # stop = 3000 #1859
    # stop = np.argmax(avg>15500)
    stop = -1
    if stop == 0:
        stop = -1
    
    if DETACHMENT_TIME_SEC == 0.03:
        ax = ax2
    else:
        ax = ax1
        
    ax.plot(times[:stop:1000], avg[:stop:1000], label = f'$E(Δt_d)$ = {DETACHMENT_TIME_SEC} s', color = colours[i],)
    ax.fill_between(times[:stop:1000], avg[:stop:1000]-stdev[:stop:1000], avg[:stop:1000]+stdev[:stop:1000],alpha=0.6, color = colours[i],)
    
for ax in [ax1,ax2]: 
    ax.set_xlabel('Time [s]', fontsize=font_mid)
    ax.set_ylabel('Thrombus size [μm$^2$]', fontsize=font_mid)
    ax.legend(fontsize=font_small)
    ax.set_xlim([0,600])
    ax.set_xticks([0,120,240,360,480,600],[0,2,4,6,8,10], fontsize=font_small)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

ax1.set_ylim([0,15000])
ax1.set_yticks([0,2500,5000,7500,10000,12500,15000], fontsize=font_small)
ax2.set_ylim([0,300])
ax2.set_yticks([0,100,200,300], fontsize=font_small)


'''###########################################################################
FIGURE 5B
##########################################################################'''

save_name = 'CAED/test detachment rate/video/Simulation 1 of full model with td = 0.03.pkl'
sample_case = pickle.load(open(save_name, 'rb'))
activation = results['final activation'][1:15, 90:170]
activation_masked = np.ma.masked_where(activation==0, activation)
activation_colormap = ax3.imshow(-1*activation_masked, origin='lower',cmap = plt.cm.cividis, vmin = -1, vmax = 0)
ax3.spines['top'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_xticks([])
ax3.set_yticks([])
colorbar_activation = plt.colorbar(activation_colormap, cax=ax3_CB)
colorbar_activation.set_ticks([0,-0.2,-0.4,-0.6,-0.8,-1])
colorbar_activation.ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1], fontsize=font_tiny)


'''############################################################################
ADD TITLES AND SAVE
############################################################################'''

ax1.set_title('A', fontsize=font_large, loc='left', y=1, fontweight='bold')
ax3.set_title('B', fontsize=font_large, loc='left', y=1, fontweight='bold')

fig.savefig(f'Figure5.{save_format}', format = save_format)