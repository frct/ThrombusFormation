# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:34:04 2024

@author: vf926215
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np


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

initial_binding_times = [0.05, 0.2, 0.5]

for i,t in enumerate(initial_binding_times):
    [avg_growth,
     stdev_growth,
     avg_growth_rate,
     stdev_growth_rate,
     times_growth,
     times,
     activation] = pickle.load(open(f'EAnoD/averages for t_bind = {t}.pkl','rb'))
    
    ax1.plot(times_growth[::step], avg_growth[::step], lw=0.5)
    ax1.fill_between(times_growth[::step], avg_growth[::step]-stdev_growth[::step], avg_growth[::step]+stdev_growth[::step],alpha=0.6, edgecolor='none')
    
    ax2.plot(times[::step], avg_growth_rate[::step],linewidth=0.5,color=colors[i % len(colors)], label=f'{int(t*1000)} ms')
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