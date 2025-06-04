# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 11:14:46 2025

@author: vf926215
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

diff = 2
n_histograms = 20

intervals = np.zeros(n_histograms)

for h in range(1,n_histograms+1):
    with open(f'exit distributions/diff = {diff}/exit distribution {h}.pkl', 'rb') as file:
        hist = pickle.load(file)
        intervals[h-1] = hist['time interval']
        t0 = hist['record start']
        t1 = t0 + hist['time interval']

        x = hist['exit distribution'][1:-1]              # Values at each index
        y = np.arange(1,len(x)+1)
        
        plt.figure()
        plt.barh(y, x, color='steelblue')
        plt.xlabel('Count')
        plt.ylabel('Exit height [μm]')
        plt.title(f'Exit distribution between {t0} and {t1}')

    
plt.figure()
plt.plot(np.arange(1,n_histograms+1), intervals)