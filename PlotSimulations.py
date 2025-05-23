# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:23:29 2025

@author: vf926215
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

files = [
    'simulation 1 of β = 0.001.pkl',
    'simulation 1 of β = 0.001 with detachment.pkl',
    'simulation 1 of β = 0.001 with detachment small D.pkl',
    'simulation 1 of β = 0.001 with detachment small D and strong margination.pkl'
    ]
titles = [
    'no detachment',
    'with detachment',
    'reduced diffusion',
    'increased margination']

for title, file in zip(titles,files):
    res = pickle.load(open(file, 'rb'))
    t = np.array([i * res['Δt'] for i in range(len(res['clot size']))])
    
    f, (ax1,ax2) = plt.subplots(2,1)
    
    ax1.plot(t, res['platelet count'])
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('platelets')
    ax1.set_xlim(0,t[100])
    ax1.set_ylim(0, res['platelet count'][-1] * 1.1)
    
    ax2.imshow(res['final density'], origin='lower')
    f.suptitle(title)
    f.tight_layout()