# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:34:04 2024

@author: vf926215
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
from RunSimulation import RunSimulation


# Get the default color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


fig = plt.figure(figsize = (5.2,2.6),dpi=900)
ax1 = fig.add_axes((0.1,0.1,0.3,0.5))
ax2 = fig.add_axes((0.6,0.1,0.3,0.5))

font_tiny = 6
font_small = 8
font_mid = 10
font_large = 12

save_format = 'svg'

step = 1000


BINDING_TIMES = [0.05, 0.2, 0.5]
n_repeats = 10
T = 600
HALF_ACTIVATION_SEC = 0.2

averages = []
errors = []


sliding_window_sec = 60



step_sec=10

for i,tb in enumerate(BINDING_TIMES):

    clot_growths = []
    attachments = []
    
    try:
        [avg_growth,
         stdev_growth,
         avg_growth_rate,
         stdev_growth_rate,
         times_growth,
         times,
         activation] = pickle.load(open(f'EAnoD/averages for t_bind = {tb}.pkl','rb'))

    except:
        for rep in range(1,n_repeats+1):
            save_name = f'EAnoD/Simulation {rep} with initial binding time = {tb}.pkl'
            
            try:
                results = pickle.load(open(save_name, 'rb'))
            except:               
                results = RunSimulation(save_file=save_name,
                                        T = T,
                                        BINDING_TIME_SEC=tb,
                                        HALF_ACTIVATION_SEC = HALF_ACTIVATION_SEC)
                
            
            Nt = len(results['clot size'])
            Nt_window = int(sliding_window_sec / results['Δt'])
            N_steps = int(step_sec / results['Δt'])
            binding_rate = [np.sum(results['binding_events'][t:t+Nt_window])/sliding_window_sec for t in range(Nt-Nt_window)]
            
            clot_growths.append(results['clot size'])
            attachments.append(binding_rate)
    
            
        times_growth = [results['Δt'] * i for i in range(len(results['clot size']))]

        avg_growth = np.mean(np.array(clot_growths), axis=0)
        stdev_growth = np.std(np.array(clot_growths), axis=0)
                
        times = [results['Δt'] * i for i in range(Nt_window, Nt)]
        avg_att = np.mean(np.array(attachments), axis=0)
        stdev_att = np.std(np.array(attachments), axis=0)
        
        pickle.dump([avg_growth,
                      stdev_growth,
                      avg_att,
                      stdev_att,
                      times_growth,
                      times,
                      results['final activation']],
                    open(f'EAnoD/averages for t_bind = {tb}.pkl','wb'))
    
    ax1.plot(times_growth[::step], avg_growth[::step], lw=0.5)
    ax1.fill_between(times_growth[::step], avg_growth[::step]-stdev_growth[::step], avg_growth[::step]+stdev_growth[::step],alpha=0.6, edgecolor='none')
    
    ax2.plot(times[::step], avg_growth_rate[::step],linewidth=0.5,color=colors[i % len(colors)], label=f'{int(tb*1000)} ms')
    ax2.fill_between(times[::step], avg_growth_rate[::step]-stdev_growth_rate[::step], avg_growth_rate[::step]+stdev_growth_rate[::step], alpha=0.6, color=colors[i % len(colors)], edgecolor='none')
    
   
ax1.set_xticks(np.arange(0,601,120), np.arange(0,11,2),fontsize=font_small)
ax1.set_yticks(np.arange(0,1501,500), np.arange(0,1501,500), fontsize=font_small)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylim([0,1500])
ax1.set_xlim([0,600])
ax1.set_xlabel('time [min]',fontsize=font_mid)
ax1.set_ylabel('thrombus size [μm$^2$]',fontsize=font_mid)
ax1.tick_params(length=1,width=0.5)


ax2.legend(fontsize=font_small) 
ax2.set_xticks(np.arange(0,601,120), np.arange(0,11,2),fontsize=font_small)
ax2.set_ylim([-0.1,3])
ax2.set_xlim([0,600])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)    
ax2.set_yticks([0,1,2,3], [0,1,2,3], fontsize=font_small)
ax2.tick_params(length=1, width=0.5)
ax2.axhline(0, linestyle = ':', color='black', lw = 0.5)
ax2.set_ylabel('platelets/s', fontsize=font_mid)
ax2.set_xlabel('time [min]', fontsize=font_mid)

'''############################################################################
ADD TITLES AND SAVE
############################################################################'''

ax1.set_title('A', fontsize=font_large, loc='left', y=1, fontweight='bold')
ax2.set_title('B', fontsize=font_large, loc='left', y=1, fontweight='bold')
            
fig.savefig(f'Figure6.{save_format}', format=save_format)