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


files = os.listdir('trajectories/no margination/')
trajectories = []

for file in files:
    with open(f'trajectories/no margination/{file}', "rb") as f:
        traj = pickle.load(f)
        trajectories.append(traj)
        
fig,ax = plt.subplots()
ax.set_xlim(0,256)
ax.set_ylim(0,64)
ax.set_aspect('equal')

num_trajectories = 154

frame_rate = 15
writer = animation.PillowWriter(fps = frame_rate)
save_file_name = 'trajectories.gif'

# Get a colormap with enough colors
cmap = plt.get_cmap('hsv')  # hsv gives a nice range of colors
colors = [cmap(i / num_trajectories) for i in range(num_trajectories)]

# initialise moving particles
x = [p[0] for p in trajectories[0]]
y = [p[1] for p in trajectories[0]]
scatter = ax.scatter(x,y, s=1, c='black')



with writer.saving(fig, save_file_name, dpi = 300):

        
        
    for frame in tqdm(range(len(trajectories))):
        ax.set_title(f'frame {frame}', fontsize=18, loc='right')
        scatter.remove()    
        x = [p[0] for p in trajectories[frame]]
        y = [p[1] for p in trajectories[frame]]
        scatter = ax.scatter(x,y,s=1,c='black')
        ax.set_aspect('equal')
        
        
        writer.grab_frame()

# # Create plot objects with assigned colors
# points = [ax.plot([], [], 'o', color=colors[i])[0] for i in range(num_trajectories)]



# def init():
#     update(0)
#     return points

# def update(frame):
#     for i, [x,y] in enumerate(trajectories[frame]):
#         points[i].set_data(x, y)
#     return points

# ani = animation.FuncAnimation(fig, update, frames=len(trajectories), init_func=init, blit=False, interval=200, repeat=False)

# plt.show()