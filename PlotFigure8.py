# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:34:58 2024

@author: vf926215
"""

import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from RunSimulation import RunSimulation

mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
save_format = 'svg'

t50 = [0, 0.2, 0.5, 1] #, 1] # 2, 5, 10] #0, 0.2, 0.5, 1,
zoom_height = 25
zoom_length = 80
zoom_ratio = zoom_length / zoom_height
zoom_start = int((256-zoom_length)/2)
zoom_end = int((256+zoom_length)/2)

# Get the default color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

PLOS_dim = (7.5,8.75)
PLOS_col = (5.2,8.75)

fig = plt.figure(figsize=PLOS_dim, dpi=300)
ax_a = fig.add_axes((0.1,0.8,0.2,0.15))
ax_bi = fig.add_axes((0.4,0.9,0.2,0.05))
ax_bii = fig.add_axes((0.65,0.9,0.2,0.05))
ax_biii = fig.add_axes((0.4,0.8,0.2,0.05))
ax_biv = fig.add_axes((0.65,0.8,0.2,0.05))
ax_b = [ax_bi, ax_bii, ax_biii, ax_biv]
ax_c = fig.add_axes((0.1,0.5,0.2,0.2))

h = 0.07
ax_di = fig.add_axes((0.4,0.6,zoom_ratio*h,h))
ax_dii = fig.add_axes((0.65,0.6,zoom_ratio*h,h))
ax_diii = fig.add_axes((0.4,0.5,zoom_ratio*h,h))
ax_div = fig.add_axes((0.65,0.5,zoom_ratio*h,h))
ax_d = [ax_di, ax_dii, ax_diii, ax_div]

step = 1000

for i,HALF_ACTIVATION_SEC in enumerate(t50):
    try:
        [avg_growth,
          stdev_growth,
          avg_growth_rate,
          stdev_growth_rate,
          avg_att,
          stdev_att,
          avg_det,
          stdev_det,
          times_growth,
          times,
          avg_density,
          stdev_density,
          activation] = pickle.load(open(f'EAED/test t50/averages for t50 = {HALF_ACTIVATION_SEC}.pkl','rb'))
    except:
        clot_growths = []
        detachments = []
        attachments = []
        net_growth = []
        density_ratio = []
        
        BINDING_TIME_SEC = 0.05 
        DETACHMENT_TIME_SEC = 0.1
        T = 600

        sliding_window_sec = 60
        start_row = 1
        
        n_repeats = 10
        
        for rep in range(1,n_repeats+1):
            save_name = f'EAED/test t50/Simulation {rep} of full model with t50 = {HALF_ACTIVATION_SEC}.pkl'
            
            try:
                results = pickle.load(open(save_name, 'rb'))
            except:               
                results = RunSimulation(save_file=save_name,
                                        T = T,
                                        BINDING_TIME_SEC=BINDING_TIME_SEC,
                                        DETACHMENT_TIME_SEC = DETACHMENT_TIME_SEC,
                                        HALF_ACTIVATION_SEC = HALF_ACTIVATION_SEC)
                
            clot_growths.append(results['clot size'])
            Nt = len(results['clot size'])
            Nt_window = int(sliding_window_sec / results['Δt'])
            Δclot = results['clot size'][1:] - results['clot size'][:-1]
            
            growth_rate = [np.sum(Δclot[t:t+Nt_window])/sliding_window_sec for t in range(Nt-Nt_window)]
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

    ax_a.plot(times_growth[::step], avg_growth[::step], lw=0.5, label=f'{HALF_ACTIVATION_SEC} s')
    ax_a.fill_between(times_growth[::step], avg_growth[::step]-stdev_growth[::step], avg_growth[::step]+stdev_growth[::step],alpha=0.6)
    
    ax_b[i].plot(times[::step], avg_growth_rate[::step],linewidth=0.5,color=colors[i % len(colors)])
    ax_b[i].fill_between(times[::step], avg_growth_rate[::step]-stdev_growth_rate[::step], avg_growth_rate[::step]+stdev_growth_rate[::step], linewidth=0.5,color=colors[i % len(colors)], alpha=0.6, edgecolor='none')
    #ax_b[i].plot(times[::step], avg_det[::step],linewidth=0.5,linestyle=':',color=colors[i % len(colors)])
    
    ax_c.bar(i,avg_density)
    ax_c.errorbar(i,avg_density, yerr=stdev_density, fmt='', elinewidth=0.5,capthick=0.5, capsize =1, ecolor='black')
    
    activation_zoom = activation[1:zoom_height, zoom_start:zoom_end]
    activation_masked = np.ma.masked_where(activation_zoom==0, activation_zoom)
    ax_d[i].imshow(-1*activation_masked, origin='lower',cmap = plt.cm.cividis, vmin = -1, vmax = 0, interpolation='None')

ax_a.legend(fontsize=8)    
ax_a.set_xticks(np.arange(0,601,120), np.arange(0,11,2),fontsize=8)
ax_a.set_yticks(np.arange(0,501,250), np.arange(0,501,250), fontsize=8)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
ax_a.set_xlim([0,600])
ax_a.set_ylim([0,500])
ax_a.set_xlabel('time [min]',fontsize=10)
ax_a.set_ylabel('thrombus size [μm$^2$]',fontsize=10)
ax_a.set_title('A', fontsize=12, loc='left', y=1, fontweight='bold')
ax_a.tick_params(length=1,width=0.5)

for i, ax in enumerate(ax_b):
    ax.set_xticks(np.arange(0,601,120), np.arange(0,11,2),fontsize=8)
    ax.set_xlim([0,600])
    ax.set_ylim([-1,5])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)    
    #ax.set_yticks([-5,0,5,10,15], [-5,0,5,10,15], fontsize=8)
    ax.tick_params(length=1, width=0.5)
    ax.axhline(0, linestyle = ':', color='black', lw = 0.5)
    
    if i == 0:
        ax.set_title('B', fontsize=12, loc='left', y=1, fontweight='bold')
        
    if i == 0:
        ax.set_ylabel('platelets/s', fontsize=10)
    if i == 2:
        ax.set_xlabel('time [min]', fontsize=10)


ax_c.set_xticks([0,1,2,3], [f'{t}' for t in t50],fontsize=8)
ax_c.set_yticks(np.linspace(0,100,6), [i*20 for i in range(6)], fontsize=8)
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
#ax_c.set_xlim([0,4])
ax_c.set_ylim([0,100])
ax_c.set_xlabel('$t_{50}$ [s]', fontsize=10)
ax_c.set_ylabel('compactness [%]', fontsize=10)
ax_c.set_title('C', fontsize=12, loc='left', y=1, fontweight='bold')
ax_c.tick_params(length=1,width=0.5)

for i, ax in enumerate(ax_d):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'$t_{{50}}$={t50[i]} s',fontsize=10)

fig.text(0.35,0.7, 'D', fontsize=12, fontweight='bold')
            
fig.savefig('Figure8.{save_format}', format=save_format)