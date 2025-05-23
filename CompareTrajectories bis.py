# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:45:20 2025

@author: vf926215
"""

'''#############################################################################

compare different scales

############################################################################'''

# scales = [1,2,5,10]
# CFL = 0.8
# y_start = [1,3,5,7,9]

# simulation = pickle.load(open('saved simulation.pkl', 'rb'))
# orig_density = simulation['density_frames'][1:-1,:,-1]

# Ny, Nx = np.shape(orig_density)

# masked_density = np.ma.masked_where(orig_density==0, orig_density)

# colors = ['blue', 'green', 'orange', 'red']

# for y in y_start:
    
#     fig, ax = plt.subplots()    
#     ax.imshow(masked_density, cmap = 'viridis', origin='lower')
    
#     if y == 1:
#         timestep = 0.01 
#     elif y < 6:
#         timestep = 0.005
#     else:
#         timestep = 0.001      
    
#     for j,scale in enumerate(scales):
#         res = pickle.load(open(f'grid tests/y = {y} scale = {scale} CFL = 0.8.pkl', 'rb'))
        
#         Nt = np.shape(res['trajectories'])[2]
#         Δt = 1 / Nt # simulations were run for 1s
#         t = np.array([i*Δt for i in range(Nt)])
#         # Calculate the indices closest to each Dt interval
#         time_intervals = np.arange(0, t[-1], timestep)
#         indices = [np.argmin(np.abs(t - Ti)) for Ti in time_intervals]
        
#         x_mean = np.median(res['trajectories'][:,0,indices], axis=0) / scale
#         y_mean = np.median(res['trajectories'][:,1,indices], axis=0) / scale
        
#         x_f_quartile = np.percentile(res['trajectories'][:,0,indices], 25, axis=0) / scale
#         x_t_quartile = np.percentile(res['trajectories'][:,0,indices], 75, axis=0) / scale
#         y_f_quartile = np.percentile(res['trajectories'][:,1,indices], 25, axis=0) / scale
#         y_t_quartile = np.percentile(res['trajectories'][:,1,indices], 75, axis=0) / scale
        
#         # Find first index where x_mean >= 180 * scale
#         cross_indices = np.where(x_mean >= 180)[0]
        
#         if cross_indices.size > 0:
#             cutoff = cross_indices[0]
#             x_mean = x_mean[:cutoff]
#             y_mean = y_mean[:cutoff]
#             x_f_quartile = x_f_quartile[:cutoff]
#             x_t_quartile = x_t_quartile[:cutoff]
#             y_f_quartile = y_f_quartile[:cutoff]
#             y_t_quartile = y_t_quartile[:cutoff]
        
#         x_err = np.vstack([x_mean - x_f_quartile, x_t_quartile - x_mean])
#         y_err = np.vstack([y_mean - y_f_quartile, y_t_quartile - y_mean])
#         ax.errorbar(x_mean, y_mean, xerr = x_err, yerr = y_err, fmt='o')
#         # Add ellipses
#         # for i in range(len(x_mean)):
#         #     width = x_t_quartile[i] - x_f_quartile[i]
#         #     height = y_t_quartile[i] - y_f_quartile[i]
#         #     ellipse = Ellipse((x_mean[i], y_mean[i]), width, height,
#         #                       edgecolor=colors[j], facecolor='none', lw=1,
#         #                       label = f'scale = {scale}' if i == 0 else None)
#         #     ax.add_patch(ellipse)
#         # ax.legend()


#     ax.set_title(f'initial y = {y}, timestep = {timestep} s')
#     ax.set_xlim([Nx // 2 - 65,Nx // 2  + 65])
#     ax.set_ylim([0,30])     
    