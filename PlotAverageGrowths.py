# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 15:13:12 2025

@author: vf926215
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

beta = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
reps = np.arange(20)

avg = np.zeros((len(beta), 19870))
error = np.zeros((len(beta), 19870))
for i,b in enumerate(beta):
    clots = np.zeros((len(reps),19870))
    for ii, rep in enumerate(reps):
        res = pickle.load(open(f'calibration/simulation {rep+1} of beta = {b:.3f} trajectories.pkl', 'rb'))
        clots[ii,:] = res['clot size']
    plt.figure()
    plt.plot(clots.T)
    avg[i,:] = np.mean(clots,axis=0)
    error[i,:] = np.std(clots, axis=0)

time = np.linspace(0, res['T'], np.shape(avg)[1])
plt.figure()
for i in range(len(beta)):
    plt.plot(time, avg[i,:], label = f'beta = {beta[i]}')
    plt.fill_between(time, avg[i,:] - error[i,:], avg[i,:] + error[i,:], alpha = 0.1)
    
plt.legend()