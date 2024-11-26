# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:46:44 2024

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

zoom_height = 25
zoom_length = 80
zoom_ratio = zoom_length / zoom_height
zoom_start = int((256-zoom_length)/2)
zoom_end = int((256+zoom_length)/2)

# Get the default color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

max_dim = (7.5,8.75)

fig = plt.figure(figsize=max_dim, dpi=900)
ax_a = fig.add_axes((0.13,0.6,0.3,0.2))
ax_bi = fig.add_axes((0.6,0.6,0.3,0.05))
ax_bii = fig.add_axes((0.6,0.7,0.3,0.05))
ax_biii = fig.add_axes((0.6,0.8,0.3,0.05))
ax_b = [ax_bi, ax_bii, ax_biii]
ax_c = fig.add_axes((0.1,0.1,0.2,0.2))

h = 0.08
ax_di = fig.add_axes((0.6,0.4,zoom_ratio*h,h))
ax_dii = fig.add_axes((0.6,0.25,zoom_ratio*h,h))
ax_diii = fig.add_axes((0.6,0.1,zoom_ratio*h,h))
ax_d = [ax_di, ax_dii, ax_diii]

save_format = 'svg'

step = 1000

BINDING_TIME_SEC = 0.05 
t_D = [0.05, 0.1, 0.2]
HALF_ACTIVATION_SEC = 0.2

n_repeats = 10
T = 1200
sliding_window_sec = 60
start_row = 1

for i,DETACHMENT_TIME_SEC in enumerate(t_D):
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
          activation] = pickle.load(open(f'EAED/test detachment rate/averages for td = {DETACHMENT_TIME_SEC}.pkl','rb'))
    except:
        clot_growths = []
        detachments = []
        attachments = []
        net_growth = []
        density_ratio =[]
        
        for rep in range(1,n_repeats+1):
            save_name = f'EAED/test detachment rate/Simulation {rep} of full model with td = {DETACHMENT_TIME_SEC}.pkl'
            
            try:
                results = pickle.load(open(save_name, 'rb'))
            except:
                results = RunSimulation(save_file=save_name,
                                        T = T,
                                        BINDING_TIME_SEC=BINDING_TIME_SEC,
                                        DETACHMENT_TIME_SEC = DETACHMENT_TIME_SEC,
                                        HALF_ACTIVATION_SEC = HALF_ACTIVATION_SEC,
                                        want_frames=True)
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
                
            times_growth = [results['Δt'] * i for i in range(len(results['clot size']))]
            
            
            avg_growth = np.mean(np.array(clot_growths), axis=0)
            stdev_growth = np.std(np.array(clot_growths), axis=0)
            
            times = [results['Δt'] * i for i in range(Nt_window, Nt)]
            
            avg_growth_rate = np.mean(np.array(net_growth), axis=0)
            stdev_growth_rate = np.std(np.array(net_growth), axis=0)
            avg_det = np.mean(np.array(detachments), axis=0)
            stdev_det = np.std(np.array(detachments), axis=0)
            avg_att = np.mean(np.array(attachments), axis=0)
            stdev_att = np.std(np.array(attachments), axis=0)
                            
            avg_density = np.mean(density_ratio) * 100
            stdev_density = np.std(density_ratio) * 100
         
            pickle.dump([avg_growth,
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
                         results['final activation']],
                        open(f'EAED/test detachment rate/averages for td = {DETACHMENT_TIME_SEC}.pkl','wb'))


    ax_a.plot(times_growth[::step], avg_growth[::step], lw=0.5, label=f'$E(\Delta t_d)= {int(DETACHMENT_TIME_SEC*1000)}$ ms')
    ax_a.fill_between(times_growth[::step], avg_growth[::step]-stdev_growth[::step], avg_growth[::step]+stdev_growth[::step],alpha=0.6, edgecolor='none')
    
        
    ax_b[i].plot(times[::step], avg_growth_rate[::step],linewidth=0.5,color=colors[i % len(colors)])
    ax_b[i].fill_between(times[::step], avg_growth_rate[::step]-stdev_growth_rate[::step], avg_growth_rate[::step]+stdev_growth_rate[::step], alpha=0.6, color=colors[i % len(colors)], edgecolor='none')
    
    ax_c.bar(i,avg_density)
    ax_c.errorbar(i,avg_density, yerr=stdev_density, fmt='', elinewidth=0.5,capthick=0.5, capsize =1, ecolor='black')
    
    activation_zoom = activation[1:zoom_height, zoom_start:zoom_end]
    activation_masked = np.ma.masked_where(activation_zoom==0, activation_zoom)
    ax_d[i].imshow(-1*activation_masked, origin='lower',cmap = plt.cm.cividis, vmin = -1, vmax = 0, interpolation='None')

ax_a.legend(fontsize=8)    
ax_a.set_xticks(np.arange(0,1201,300), np.arange(0,21,5),fontsize=8)
ax_a.set_yticks(np.arange(0,1001,250), np.arange(0,1001,250), fontsize=8)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
ax_a.set_xlim([0,1200])
ax_a.set_ylim([0,1000])
ax_a.set_xlabel('time [min]',fontsize=10)
ax_a.set_ylabel('thrombus size [μm$^2$]',fontsize=10)
ax_a.set_title('A', fontsize=12, loc='left', y=1, fontweight='bold')
ax_a.tick_params(length=1,width=0.5)

for i, ax in enumerate(ax_b):
    ax.set_xticks([])
    ax.set_xlim([0,1200])
    ax.set_ylim([-1,6])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)    
    #ax.set_yticks([-10,0,10,20,30], [-10,0,10,20,30], fontsize=8)
    ax.tick_params(length=1, width=0.5)
    ax.axhline(0, linestyle = ':', color='black', lw = 0.5)
    ax.set_xticks(np.arange(0,1201,300), np.arange(0,21,5),fontsize=8)
    if i == 2:
        ax.set_title('B', fontsize=12, loc='left', y=1, fontweight='bold')
        
    if i == 1:
        ax.set_ylabel('platelets/s', fontsize=10)
        # ax.legend(handles=[
        #     plt.Line2D([], [], color='black', label='Attachment'),
        #     plt.Line2D([], [], linestyle='--', color='black', label='Detachment')
        # ], loc='upper right', fontsize=8)
    if i == 0:        
        ax.set_xlabel('time [min]', fontsize=10)


ax_c.set_xticks([0,1,2], [f'{round(t*1000)}' for t in t_D],fontsize=8)
ax_c.set_yticks(np.linspace(0,100,6), [i*20 for i in range(6)], fontsize=8)
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
#ax_c.set_xlim([0,4])
ax_c.set_ylim([0,100])
ax_c.set_xlabel('$E(Δt_d)$ [ms]', fontsize=10)
ax_c.set_ylabel('compactness [%]', fontsize=10)
ax_c.set_title('C', fontsize=12, loc='left', y=1, fontweight='bold')
ax_c.tick_params(length=1,width=0.5)

for i, ax in enumerate(ax_d):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'$E(Δt_d)$={round(t_D[i]*1000)} ms',fontsize=10)

'''############################################################################
ADD TITLES AND SAVE
############################################################################'''

           
fig.savefig(f'Figure7.{save_format}', format=save_format)