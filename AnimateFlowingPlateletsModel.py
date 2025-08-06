# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 07:54:01 2025

@author: vf926215
"""

import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from tqdm import tqdm
from matplotlib.colors import ListedColormap


        
fig,ax = plt.subplots(figsize=(18,5))
ax.set_xlim(0,256)
ax.set_ylim(0,64)
ax.set_aspect('equal')

vessel_walls = np.zeros((64,256))
vessel_walls[0,:] = 1
vessel_walls[-1,:] = 1
vessel_walls[0,103:153] = 2 # injury patch

custom_colors = ['white', 'tan', 'orange']  # Colors corresponding to values 1 and 2
custom_cmap = ListedColormap(custom_colors)


frame_rate = 15
writer = animation.PillowWriter(fps = frame_rate)
save_file_name = 'flowing platelets.gif'

frames = np.arange(0,150)

with writer.saving(fig, save_file_name, dpi = 300):

    vessel_walls_cm = ax.imshow(vessel_walls, cmap=custom_cmap, extent=[0, 256, 0, 64], origin='lower')   
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
        
    for frame in tqdm(frames):
        
        with open(f'frames/frame {frame}.pkl', "rb") as file:
            res = pickle.load(file)
            activation = res['activation']
            
            
        ax.set_title(f'frame {frame}', fontsize=18, loc='right')
        
        mask = (activation == 0) | (vessel_walls > 0)
        Z = np.ma.masked_where(mask, activation)
        cs = ax.imshow(-1*Z, origin ='lower', cmap = plt.cm.cividis, vmin =-1, vmax = 0, extent=[0, 256, 0, 64])
        ax.set_aspect('equal')
    
        writer.grab_frame() 
        
        cs.remove()