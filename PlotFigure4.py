# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:09:47 2024

@author: vf926215
"""


import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


max_dim = (7.5,6)

fig = plt.figure(figsize = max_dim, dpi=900)

ax1 = fig.add_axes((0.1,0.6,0.4,0.2))
ax2 = fig.add_axes((0.6,0.6,0.4,0.2))
ax3 = fig.add_axes((0.1,0.1,0.4,0.3))
ax4 = fig.add_axes((0.6,0.1,0.4,0.3))

font_tiny = 6
font_small = 8
font_mid = 10
font_large = 12

save_format = 'svg'

'''##########################################################
FIGURE 4A IS ADDED BY HAND IN INKSCAPE
##########################################################'''

'''##########################################################
FIGURE 4B IS MADE UP OF FRAMES OF SIMULATIONS IMPORTED
SEPARATELY INTO INKSCAPE AND OF AN AVERAGE ACTIVATION.
ONLY THE LATTER IS SHOWN HERE
##########################################################'''

injury = np.arange(103,154)
offset = 5


j_samples = np.arange(injury[0]+offset,injury[-1]-offset) # np.arange(103,153) #[108,118,128,138,148]

N_reps = 10
colours = ['blue', 'orange', 'green','red', 'cyan']

frames = [6, 12, 25, 50, 100]

rows = []
all_cols = []

for i, f in enumerate(frames):

    
    activation_frames = []
    
    for rep in range(1,N_reps+1):
         
        save_name = f'EAED/test detachment rate/with frames/Simulation {rep} of full model with td = 0.2.pkl'
        results = pickle.load(open(save_name, 'rb'))      
        Δt = results['Δt frame']    
            
        activation = results['activation_frames'][:,:,f]
        Ny,Nx = np.shape(activation)
        
        activation_frames.append(activation)
    
    average_activation = np.sum(np.array(activation_frames),axis=0) / N_reps
    
    average_masked = np.ma.masked_where(average_activation[1:20,90:165]==0,average_activation[1:20,90:165])
    
    if f == frames[-1]:
        ax2.imshow(-1*average_masked, origin='lower', cmap = plt.cm.cividis, vmin=-1, vmax = 0, interpolation='None')
    
    first_gap = np.where(average_activation[1:,j_samples]==0)[0][0] + 1 # add 1 because we are looking in average_activation starting from row 1           
    for distance in range(1,first_gap):
        rows.append({
            'time': int(f * Δt),
            'absolute_distance': (distance-1), 
            'relative_distance' : (distance-1) / (first_gap-1),
            'all columns' : average_activation[distance,j_samples],
            'activation': np.nanmean(average_activation[distance,j_samples]),
            'standard dev': np.std(average_activation[distance,j_samples],ddof=1),
            'standard error':  np.std(average_activation[distance,j_samples],ddof=1) / np.sqrt(len(j_samples))})
        
df = pd.DataFrame(rows)

n_samples = len(j_samples)

'''##########################################################
FIGURE 4C AND D
##########################################################'''
  
for i,f in enumerate(frames):
    df_sub = df[df['time'] == int(f * Δt)]
    ax3.plot(df_sub['absolute_distance'],
            df_sub['activation'],
            marker='o', ls='-', color = colours[i], label = f't={int(f * Δt)} s')
   
    ax4.plot(df_sub['relative_distance'],
            df_sub['activation'],
            marker='o', ls='-',color = colours[i], label = f't={int(f * Δt)} s')

ax3.set_xlim([0,14])
ax3.set_xlabel('distance [$\mu m$]', fontsize=font_mid)
ax3.set_xticks(np.arange(0,16,2), np.arange(0,16,2), fontsize=font_small)

ax4.set_xlim([0,1])
ax4.set_xlabel('relative distance', fontsize=font_mid)
ax4.set_xticks(np.arange(0,1.1,0.2), [0,0.2,0.4,0.6,0.8,1],fontsize=font_small)
ax4.plot(np.arange(2), np.arange(1,-1,-1),'black')

for ax in [ax3,ax4]:
    ax.set_yticks([0, 0.2,0.4,0.6,0.8,1],[0, 0.2,0.4,0.6,0.8,1],fontsize=font_small)         
    ax.set_ylabel('activation [A.U.]',fontsize=font_mid)
    ax.set_ylim([0,1])    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=font_small)
   
'''############################################################################
ADD TITLES AND SAVE
############################################################################'''

ax1.set_title('A', fontsize=font_large, loc='left', y=1, fontweight='bold')
ax2.set_title('B', fontsize=font_large, loc='left', y=1, fontweight='bold')
ax3.set_title('C', fontsize=font_large, loc='left', y=1, fontweight='bold')
ax4.set_title('D', fontsize=font_large, loc='left', y=1, fontweight='bold')

fig.savefig(f'Figure4.{save_format}', format = save_format)