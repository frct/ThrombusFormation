# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 11:06:47 2025

@author: vf926215
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.optimize import minimize

Ny = 62
diff = 0.005
n_histograms = 100
n_plots = 10
plot_interval = n_histograms // n_plots

intervals = np.zeros(n_histograms)
a = np.zeros(n_histograms)

nll_β = np.zeros(n_histograms)



for h in range(n_histograms):
    with open(f'y distribution/diff={diff}/platelet positions {h}.pkl', 'rb') as file:
        positions = pickle.load(file)
        
        y_normed = [(pos[1]-1)/Ny for pos in positions if (pos[1]-1)/Ny < 1] # get vertical coordinate starting from vessel wall
        
        # fit symetric beta distribution by defining negative log-likelihood and minimizing

        def neg_log_likelihood(alpha):
            if alpha <= 0:  # invalid parameter
                return np.inf
            return -np.sum(beta.logpdf(y_normed, alpha, alpha, loc=0, scale=1))

        res = minimize(neg_log_likelihood, x0=1.0, bounds=[(1e-6, None)])
        
        a[h] = res.x[0]
        nll_β[h] = res.fun   
        
        
        if h % plot_interval == 0:
            fig, ax = plt.subplots()
            # Create horizontal histogram
            ax.hist(y_normed, bins=62, orientation='horizontal', density=True, edgecolor='black', align='mid')
            x_fit = np.linspace(0, 1, 100)
            pdf_β = beta.pdf(x_fit, a[h], a[h], loc=0, scale=1)
            ax.plot(pdf_β, x_fit, 'r-', lw=2, label='β fit')
            ax.legend()
            
            
            ax.set_xlabel('Frequency (normalised by bin width)')
    
            ax.set_ylabel('Exit height (normalised)')
            
            fig.suptitle(f'Vertical distribution at {h} s')
            fig.tight_layout()

    
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))

T = np.arange(1,n_histograms+1) * 0.5

ax1.plot(T, a, '-o', label='a')
ax1.set_title('Beta distribution parameters')
ax1.legend()

ax2.plot(T,nll_β, '-o', label='neg LL of beta')
ax2.legend()

for ax in [ax1, ax2]:
    ax.set_xticks(np.linspace(0,10,4).astype(int))
    ax.set_xlabel('T save [s]')
    ax.set_xlim([T[0],T[-1]])

fig.tight_layout()

fig.savefig(f'vertical distributions for diffusion={diff}.png')