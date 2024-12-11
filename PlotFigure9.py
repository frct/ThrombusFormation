# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:10:25 2024

@author: vf926215
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
#from RunSimulation import GetExpectedBindingTime, GetExpectedDetachmentTime
import pandas as pd

font_tiny = 6
font_small = 8
font_mid = 10
font_large = 12

save_format = 'svg'

'''##############################################################
LOAD EXPERIMENTAL DATA
##############################################################'''

excel_file = 'experimental averages.xlsx' 

# Read excel file  
df = pd.read_excel(excel_file)

cutoff_fast = 267
    
fast_average = df['fast_avg']
slow_average = df['slow_avg']
fast_std = df['fast_stdDev']
slow_std = df['slow_stdDev']

'''###############################################################
LOAD SIMULATED DATA
##############################################################'''

res = pickle.load(open('optimal_simulation.pkl', 'rb'))

sim_fast = np.mean(res['fast simulation'], axis=0)
sim_slow = np.mean(res['slow simulation'], axis = 0)
std_fast = np.std(res['fast simulation'], axis=0)
std_slow = np.std(res['slow simulation'], axis=0)

data_to_save = [df['Time (s)'],
                pd.Series(sim_fast[:cutoff_fast], name = 'fast average'),
                pd.Series(std_fast[:cutoff_fast], name = 'fast stdDev'),
                pd.Series(sim_slow, name = 'slow average'),
                pd.Series(std_slow, name = 'slow stdDev')]

df2 = pd.concat(data_to_save,axis=1)
df2.columns = ['Time (s)', 'fast_avg', 'fast_stdDev', 'slow_avg', 'slow_stdDev']
df2.to_excel('optimised simulations.xlsx')



zoom=15

f = plt.figure(figsize=(3,3),dpi=900)
ax1 = f.add_axes((0.1,0.1,0.8,0.8))

ax1.fill_between(df['Time (s)'][:cutoff_fast], fast_average[:cutoff_fast]+fast_std[:cutoff_fast], fast_average[:cutoff_fast]-fast_std[:cutoff_fast], color= 'red', alpha = 0.3, edgecolor='none',zorder=1)
ax1.plot(df['Time (s)'][:cutoff_fast], fast_average[:cutoff_fast], 'red', label = 'fast responders (exp)', lw=1,zorder=1)
ax1.fill_between(df['Time (s)'][:cutoff_fast], sim_fast[:cutoff_fast]+std_fast[:cutoff_fast], sim_fast[:cutoff_fast]-std_fast[:cutoff_fast], color= 'white', alpha = 0.5, edgecolor='none',zorder=2)
ax1.plot(df['Time (s)'][:cutoff_fast], sim_fast, color='red', linestyle='--', lw = 1, label = 'fast responders (sim)', zorder=2)

ax1.plot(df['Time (s)'], slow_average, 'blue', label = 'slow responders (exp)', lw = 1,zorder=1)
ax1.fill_between(df['Time (s)'], slow_average-slow_std, slow_average+slow_std, color= 'blue', alpha = 0.3, edgecolor='none',zorder=1)
ax1.fill_between(df['Time (s)'], sim_slow-std_slow, sim_slow+std_slow, color= 'white', alpha = 0.5, edgecolor='none',zorder=2)
ax1.plot(df['Time (s)'], sim_slow, color='blue', linestyle='--', lw = 1, label = 'slow responders (sim)',zorder=2)

ax1.legend(fontsize=font_small)
ax1.set_xlabel('time [min]', fontsize=font_mid)
ax1.set_ylabel('fluorescence intensity [AU]', fontsize=font_mid)
ax1.set_xlim([0,600])
ax1.set_ylim([0,550])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xticks(np.arange(0,601,120),np.arange(0,11,2),fontsize=font_small)
ax1.set_yticks(np.arange(0,600,100),np.arange(0,600,100),fontsize=font_small)
ax1.legend(handles=[
            plt.Line2D([], [], color='black', label='experiment'),
            plt.Line2D([], [], linestyle='--', color='black', label='simulation')
        ],
        loc='upper left', fontsize=font_small)


f.savefig(f'Figure9.{save_format}', format=save_format)