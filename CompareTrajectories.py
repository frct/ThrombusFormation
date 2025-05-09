# -*- coding: utf-8 -*-
"""
Created on Thu May  8 14:44:15 2025

@author: vf926215
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


scale = 1
CFL = [0.2, 0.5, 0.8]
y_start = [1,3,5,7,9]

simulation = pickle.load(open('saved simulation.pkl', 'rb'))
density = simulation['density_frames'][1:-1,:,-1]
masked_density = np.ma.masked_where(density==0, density)

colors = ['blue', 'green', 'red']


for y in y_start:
    
    fig, ax = plt.subplots()    
    ax.imshow(masked_density, cmap = 'viridis', origin='lower')
    
    if y == 1:
        timestep = 0.01 
    elif y < 6:
        timestep = 0.005
    else:
        timestep = 0.001      
    
    for j,c in enumerate(CFL):
        res = pickle.load(open(f'grid tests/y = {y} scale = {scale} CFL = {c}.pkl', 'rb'))
        
        Nt = np.shape(res['trajectories'])[2]
        Δt = 1 / Nt # simulations were run for 1s
        t = np.array([i*Δt for i in range(Nt)])
        # Calculate the indices closest to each Dt interval
        time_intervals = np.arange(0, t[-1], timestep)
        indices = [np.argmin(np.abs(t - Ti)) for Ti in time_intervals]
        
        x_mean = np.mean(res['trajectories'][:,0,indices], axis=0)
        y_mean = np.mean(res['trajectories'][:,1,indices], axis=0)
        
        x_f_quartile = np.percentile(res['trajectories'][:,0,indices], 25, axis=0)
        x_t_quartile = np.percentile(res['trajectories'][:,0,indices], 75, axis=0)
        y_f_quartile = np.percentile(res['trajectories'][:,1,indices], 25, axis=0)
        y_t_quartile = np.percentile(res['trajectories'][:,1,indices], 75, axis=0)
        
        # Find first index where x_mean >= 180
        cross_indices = np.where(x_mean >= 180)[0]
        
        if cross_indices.size > 0:
            cutoff = cross_indices[0]
            x_mean = x_mean[:cutoff]
            y_mean = y_mean[:cutoff]
            x_f_quartile = x_f_quartile[:cutoff]
            x_t_quartile = x_t_quartile[:cutoff]
            y_f_quartile = y_f_quartile[:cutoff]
            y_t_quartile = y_t_quartile[:cutoff]
        
        # Add ellipses
        for i in range(len(x_mean)):
            width = x_t_quartile[i] - x_f_quartile[i]
            height = y_t_quartile[i] - y_f_quartile[i]
            ellipse = Ellipse((x_mean[i], y_mean[i]), width, height,
                              edgecolor=colors[j], facecolor='none', lw=1,
                              label = f'CFL = {c}' if i == 0 else None)
            ax.add_patch(ellipse)
        ax.legend()


    ax.set_title(f'initial y = {y}, timestep = {timestep} s')
    ax.set_xlim([65,175])
    ax.set_ylim([0,30])     