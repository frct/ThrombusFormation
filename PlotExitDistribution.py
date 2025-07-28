# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 11:14:46 2025

@author: vf926215
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import gamma, beta

diff = 'beta'
n_histograms = 40

intervals = np.zeros(n_histograms)
a = np.zeros(n_histograms)
b = np.zeros(n_histograms)
shape = np.zeros(n_histograms)
scale = np.zeros(n_histograms)
nll_β = np.zeros(n_histograms)
nll_γ = np.zeros(n_histograms)

for h in range(1,n_histograms+1):
    with open(f'exit distributions/diff = {diff}/exit distribution {h}.pkl', 'rb') as file:
        hist = pickle.load(file)
        intervals[h-1] = hist['time interval']
        t0 = hist['record start']
        t1 = t0 + hist['time interval']

        x = hist['exit distribution'][1:-1].astype(int)
        Ny = len(x)
        
        # fit beta distribution
        
        step = 1 / len(x)
        start = step / 2
        y_normed = np.arange(start,1,step) # the normalised y coordinates at which particles exit
        # create an array of samples of exit indices based on the counts
        #  i.e. if x_mirrored = [3, 0, 1, 2, etc] and y = [0.05, 0.15, 0.25, etc.]
        # then samples = [0.05, 0.05, 0.05, 0.25, 0.35, 0.35 ...]
        samples_β = np.repeat(y_normed, x)
        params_β = beta.fit(samples_β, floc=0, fscale=1) 
        a[h-1] = params_β[0]
        b[h-1] = params_β[1]
        
        nll_β[h-1] = -np.sum(beta.logpdf(samples_β, *params_β))

        
        # fit gamma distribution
        
        x_mirrored = x[:Ny//2] + x[-1 : Ny//2-1 : -1]
        y = np.arange(0.5,Ny//2,1)
        samples_γ = np.repeat(y, x_mirrored)
        params_γ = gamma.fit(samples_γ, floc=0)
        shape[h-1], loc, scale[h-1] = params_γ
        
        nll_γ[h-1] = -np.sum(gamma.logpdf(samples_γ, *params_γ))
        
        fig, (ax1, ax2) = plt.subplots(1,2)
        frequencies = x / np.sum(x)
        frequencies_density = frequencies / step
        ax1.barh(y_normed, frequencies_density, height = step, color='steelblue')
        x_fit = np.linspace(0, 1, 100)
        pdf_β = beta.pdf(x_fit, *params_β)
        ax1.plot(pdf_β, x_fit, 'r-', lw=2, label='β fit')
        ax1.legend()
        
        frequencies = x_mirrored/np.sum(x_mirrored)
        frequencies_density = frequencies / 1 # in the case of gamma, the bin width is exactly 1
        ax2.barh(y, frequencies_density, height=1, color='steelblue')
        x_fit = np.linspace(0,Ny//2,100)
        pdf_γ = gamma.pdf(x_fit, *params_γ)
        ax2.plot(pdf_γ, x_fit, 'r-', lw=2, label='γ fit')
        ax2.legend()
        
        ax1.set_xlabel('Frequency (normalised by bin width)')
        ax2.set_xlabel('Frequency')
        ax1.set_ylabel('Exit height (normalised)')
        ax2.set_ylabel('Exit height [μm]')
        
        fig.suptitle(f'Exit distribution between {t0} s and {t1} s')
        fig.tight_layout()
        #plt.xlim([0,700])

    
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12,4))

T = np.arange(1,n_histograms+1) * 0.5

ax1.plot(T, intervals, '-o')
ax1.set_title('Time intervals to N particles')
ax1.set_ylabel('time [s]')

ax2.plot(T, a, '-o', label='a')
ax2.plot(T, b, '-o', label='b')
ax2.set_title('Beta distribution parameters')
ax2.legend()

ax3.plot(T, scale, '-o', color = 'blue', label='scale')
ax3.set_ylabel('scale', color='blue')

ax4 = ax3.twinx()
ax4.plot(T, shape, '-o', color = 'red', label='shape')
ax4.set_ylabel('shape', color='red')

ax3.set_title('Gamma distribution parameters')

for ax in [ax1, ax2, ax3]:
    ax.set_xticks(np.linspace(2,10,4).astype(int))
    ax.set_xlabel('T save [s]')
    ax.set_xlim([T[0],T[-1]])

fig.tight_layout()

fig.savefig(f'Exit distributions for diffusion={diff}.png')

f,ax = plt.subplots()
ax.plot(T,nll_β, '-o', label='neg LL of beta')
ax.plot(T,nll_γ, '-o', label='neg LL of gamma')
ax.legend()
ax.set_xticks(np.linspace(2,10,4).astype(int))
ax.set_xlabel('T save [s]')
ax.set_xlim([T[0],T[-1]])
f.savefig(f'negative log-likelihoods for diffusion={diff}.png')