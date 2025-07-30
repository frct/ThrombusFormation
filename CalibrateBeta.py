# -*- coding: utf-8 -*-
"""
Created on Tue May 13 12:47:21 2025

run simulations with a range of beta values to try and determine
a value that results in an expected binding time of 50ms

@author: vf926215
"""

import matplotlib.pyplot as plt
import numpy as np
from RunSimulation import RunSimulation

beta_values = [0.1] #np.arange(0.01, 0.1, 0.01)
n_reps = 1
T = 600

for i, β in enumerate(beta_values):
    for rep in range(n_reps):
        #save_name = f'calibration/simulation {rep+1} of β = {β} trajectories with sigma =0.05.pkl'
        save_name = f'simulation of β = {β}.pkl'
        res = RunSimulation(save_name, T=T, BETA = β, DETACHMENT_TIME_SEC=0.1, flow_dependence=True, want_frames=True)
        plt.figure()
        t = np.array([i * res['Δt'] for i in range(len(res['clot size']))])
        plt.plot(t, res['clot size'])
        
        plt.figure()
        plt.imshow(res['final density'], origin='lower')