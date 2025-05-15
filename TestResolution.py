# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 10:18:53 2025

test effect of grid spacing and ts on trajectories of vesicles or platelets in
a vessel with a thrombus in the middle

@author: vf926215
"""

import numpy as np
from mpi4py import MPI
import pickle
from LBM_functions import InitialiseLBM, UpdateLBM
from PlateletModel import DriftPlatelets

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

'''###########################################################################

TUNABLE PARAMETERS

###########################################################################'''


# choose zoom which must be an integer, fyi original grid spacing is 1um
scale = 10
# choose CFL which will determine the ts
CFL = 0.8

T = 1
particle_R = 1e-6 # radius of particles which will determine diffusivity
kB = 1.38e-23
Temp = 310
N_reps = 100


'''###########################################################################

GENERATE SIMULATION DOMAIN

############################################################################'''

simulation = pickle.load(open('saved simulation.pkl', 'rb'))

orig_density = simulation['density_frames'][1:-1,:,-1]
orig_Δx = 1e-6

density = np.repeat(np.repeat(orig_density, scale, axis=0), scale, axis=1)
density = np.pad(density, ((1,1),(0,0)), mode='constant',constant_values=1)
porosity = 1 - density

Ny, Nx = np.shape(density)

'''###########################################################################

GET THE VELOCITY FIELD

###########################################################################'''

Δx_USI = orig_Δx / scale # lattice unit (lu) size in m
C_ρ = 1000 # conversion factor for ρ in kg/m3
ρ_USI = 1060 # kg/m3
RADIUS_USI = (Ny-2) * Δx_USI / 2 # m
γ_USI = 1000 # s-1
U_MAX_USI = γ_USI * RADIUS_USI / 2 # m/s
μ_USI = 4e-3 # extrapolation from Cherry 2013 and lab calculation for in vitro experiments
NU_USI = μ_USI / ρ_USI
τ = 0.809 # dimensionless characteristic relaxation time, must be bigger than 0.5, ideal value = 0.809


Cs, LBM_umax, nu, Δt_LBM, ρ0, dP_dx, F, ρ, ux, uy = InitialiseLBM(Nx, Ny, Δx_USI, τ, NU_USI, ρ_USI, U_MAX_USI, C_ρ)

try:
    [ux,uy] = pickle.load(open(f'Velocity field for scale = {scale}.pkl', 'rb'))
except:        
    F = np.einsum('ijk,ij->ijk', F, porosity)
    F, ux, uy, vel, ρ, *_ = UpdateLBM(porosity, F, ρ0, τ, dP_dx, Cs, N_convergence=100)
    pickle.dump([ux,uy], open(f'Velocity field for scale = {scale}.pkl', 'wb'))

'''###########################################################################

ADD PARTICLES AND TRACK TRAJECTORY

############################################################################'''

y_start = 9.5

Δt = CFL / np.max(np.sqrt(ux**2 + uy**2)) * Δt_LBM
Nt = int(T / Δt)
D = kB * Temp / (6 * np.pi * μ_USI * particle_R)
σ_diffusion = np.sqrt(2 * D * Δt / Δx_USI**2)

trajectories = np.zeros((N_reps,2,Nt))

for repeat in range(N_reps):
    print(f'Rep {repeat}')
    particle = [[Nx//4, y_start * scale]]

    for t in range(Nt):
        trajectories[repeat,:,t] = particle[0]
        particle = DriftPlatelets(particle, ux, uy, Nx-1, Ny-1, Δt/Δt_LBM, σ_diffusion)
    
pickle.dump({'trajectories':trajectories, 'scale':scale, 'CFL': CFL, 'σ_diffusion': σ_diffusion, 'D': D}, open(f'y = {int(y_start)} scale = {scale} CFL = {CFL}.pkl', 'wb'))