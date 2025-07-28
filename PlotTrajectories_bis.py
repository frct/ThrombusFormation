# -*- coding: utf-8 -*-
"""
Created on Wed May 28 11:50:35 2025

@author: vf926215
"""

import os
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
save_file_name = 'trajectories with strong binding.gif'



with writer.saving(fig, save_file_name, dpi = 300):

    vessel_walls_cm = ax.imshow(vessel_walls, cmap=custom_cmap, extent=[0, 256, 0, 64], origin='lower')   
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
        
    for frame in tqdm(np.arange(400,601)):
        
        with open(f'trajectories/platelet positions {frame}.pkl', "rb") as file:
            [platelets, density, activation, t] = pickle.load(file)
        
        ax.set_title(f'frame {frame}', fontsize=18, loc='right')
          
        x = [p[0] for p in platelets]
        y = [p[1] for p in platelets]
        scatter = ax.scatter(x,y,s=1,c='black')
        
        mask = (activation == 0) | (vessel_walls > 0)
        Z = np.ma.masked_where(mask, activation)
        cs = ax.imshow(-1*Z, origin ='lower', cmap = plt.cm.cividis, vmin =-1, vmax = 0, extent=[0, 256, 0, 64])
        ax.set_aspect('equal')
    
        writer.grab_frame()
        
        scatter.remove()  