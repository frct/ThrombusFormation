# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:07:50 2023

@author: vf926215
"""

import numpy as np

## define global variables for D2Q9 scheme (plasma)
Nc = 9
idxs = np.arange(Nc)
cxs = np.array([0, 1, 1, 0, -1, -1, -1,  0,  1])
cys = np.array([0, 0, 1, 1,  1,  0, -1, -1, -1])
weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36]) # fyi sums to 1
FGlobals = [Nc, idxs, cxs, cys, weights]

def InitialiseLBM(Nx, Ny, Δx, τ, NU, RHO, Peak_U, C_ρ=1000):
    ''' Given dimensions and physical quantities, initialises the LBM quantities
    and f function in a straight vessel without obstacles to match Poiseuille
    by assuming that f is at equilibrium'''
    
    
    ### FIXED DIMENSIONLESS LBM PARAMETERS
    
    Cs = 1 / np.sqrt(3) # celerity of sound in lu/ts
    LBM_umax = Cs * 0.1
    nu = Cs**2 * (τ-1/2)


    ### CONVERSION FACTORS
    
    C_nu = NU / nu
    Δt = Δx**2 / C_nu
    C_u = Δx / Δt

    ### LAST DIMENSIONLESS PARAMETERS
    
    ρ0 = RHO / C_ρ
    peak_u = Peak_U / C_u
    dP = 2 * nu * ρ0 * peak_u / ((Ny-2)/2)**2
    
    ### initialise velocity as Poiseuille

    ux = np.zeros((Ny,Nx))
    uy = np.zeros((Ny,Nx))

    R = (Ny-2) / 2 # 2 outer rows correspond to vessel wall
    for idy in range(1,Ny-1):
        y = (idy-1) + 1 / 2
        r = y - R
        ux[idy,:] = peak_u * (1-r**2/R**2)
    
    ρ = np.zeros_like(ux)
    ρ[1:-1,:] = ρ0
    F = GetFeq(ρ, ux, uy)
    
    return Cs, LBM_umax, nu, Δt, ρ0, dP, F, ρ, ux, uy

def GetFeq(ρ, ux=0, uy=0):
    Nc, idxs, cxs, cys, weights = FGlobals 
    Ny, Nx = np.shape(ρ)
    Feq = np.zeros((Ny, Nx, Nc))
    for i, (w,cx,cy) in enumerate(zip(weights,cxs,cys)):
        Feq[:,:,i] = w * ρ * (1 + 3*(cx*ux+cy*uy) + 9/2*(cx*ux+cy*uy)**2 - 3/2*(ux**2+uy**2))
    return Feq

def ExtractMacro(F):
    ''' Returns macroscopic quantities ux, uy, density and pressure (all in 
    dimensionless units) from the LBM f function'''
    
    Ny,Nx, Nc = np.shape(F)
    ρ = np.sum(F,2)
    P = ρ / 3
    ux = np.zeros((Ny,Nx))
    uy  = np.zeros((Ny,Nx))
    
    idys, idxs = np.where(ρ>1e-5)
    
    for idy, idx in zip(idys, idxs):
        ux[idy,idx] = (F[idy,idx,1] - F[idy,idx,5] + F[idy,idx,2] - F[idy,idx,6] + F[idy,idx,8] - F[idy,idx,4]) / ρ[idy,idx]
        uy[idy,idx] = (F[idy,idx,3] - F[idy,idx,7] + F[idy,idx,2] - F[idy,idx,6] + F[idy,idx,4] - F[idy,idx,8]) / ρ[idy,idx]
        
    return ρ, P, ux, uy

def UpdateLBM(porosity, F, ρ0, τ, dP_dx, CELERITY_OF_SOUND_LBM, N_convergence=10, is_print=False):
    ''' Given a newly updated F, runs the LBM until convergence is reached.
    Convergence is defined as at least 10 consecutive steps in which the 
    update difference in total velocity relative to max velocity is below 0.1%
    everywhere.'''


    ρ, _, ux, uy = ExtractMacro(F)
    vel = np.sqrt(ux**2, uy**2)
      
    t = 0    
    
    # diff_rel = []
    # diff_abs = []
    
    diff_max = []
    t_converged = 0
    
    while t_converged < N_convergence:    
        F_prev = F.copy()
        ux_prev = ux.copy()
        uy_prev = uy.copy()
        ρ_prev = ρ.copy()
        vel_old = vel.copy()
        
        F_post_collision = RelaxF(F_prev, ux_prev, uy_prev, ρ_prev, τ)        
        F_post_drift = DriftF(F_post_collision, porosity)
        F[1:-1,:,:] = GetZhangBoundaries(F_post_drift[1:-1,:,:], ρ_prev[1:-1,:], ρ0, dP_dx, CELERITY_OF_SOUND_LBM)
        ρ, _, ux, uy = ExtractMacro(F)
        vel = np.sqrt(ux**2 + uy**2)
   
        diff = np.abs(vel-vel_old)
        diff_max.append(np.nanmax(diff / np.max(vel_old)))
        
        # diff_abs.append(np.nanmax(diff))
        # diff_rel.append(np.nanmax(diff / vel_old))
        
        if diff_max[-1] < 0.001:
            t_converged +=1
        else:
            t_converged = 0
            
        t += 1
    
    if is_print:
        print(f'{t} steps till LBM convergence') 
        
    return F, ux, uy, vel, ρ, diff_max



def RelaxF(F,ux,uy,ρ,τ):
    Feq = GetFeq(ρ, ux, uy)
    newF = F - 1 / τ * (F - Feq)
    return newF


# def DriftF(F,porosity):
#     '''Pushes the F distribution along their respective axes with a partial 
#     bounce-back against cells of smaller porosity than the incoming cell'''
    
#     newF = np.zeros_like(F)
    
#     # is porosity in each of the 8 directions smaller than inflowing cell ?
    
#     porosity_1 = (porosity[:,1:] < porosity[:,:-1])
#     porosity_2 = (porosity[1:,1:] < porosity[:-1,:-1])
#     porosity_3 = (porosity[1:,:] < porosity[:-1,:])
#     porosity_4 = (porosity[1:,:-1] < porosity[:-1,1:])
#     porosity_5 = (porosity[:,:-1] < porosity[:,1:])
#     porosity_6 = (porosity[:-1,:-1] < porosity[1:,1:])
#     porosity_7 = (porosity[:-1,:] < porosity[1:,:])
#     porosity_8 = (porosity[:-1,1:] < porosity[1:,:-1])
    
    
    
#     newF[:,:,0] = F[:,:,0]
#     newF[:,1:,1] = F[:,:-1,1] * (1 - (1 - porosity[:,1:]) * porosity_1) \
#                 + (1-porosity[:,:-1]) * F[:,1:,5] * porosity_5
#     newF[1:,1:,2] = F[:-1,:-1,2] * (1 - (1 - porosity[1:,1:]) * porosity_2) \
#                 + (1-porosity[:-1,:-1]) * F[1:,1:,6] * porosity_6
#     newF[1:,:,3] = F[:-1,:,3] * (1 - (1 - porosity[1:,:]) * porosity_3) \
#                 + (1-porosity[:-1,:]) * F[1:,:,7] * porosity_7
#     newF[1:,:-1,4] = F[:-1,1:,4] * (1 - (1 - porosity[1:,:-1]) * porosity_4) \
#                 + (1-porosity[:-1,1:]) * F[1:,:-1,8] * porosity_8
#     newF[:,:-1,5] = F[:,1:,5] * (1 - (1 - porosity[:,:-1]) * porosity_5) \
#                 + (1-porosity[:,1:]) * F[:,:-1,1] * porosity_1
#     newF[:-1,:-1,6] = F[1:,1:,6] * (1 - (1 -porosity[:-1,:-1]) * porosity_6) \
#                 + (1-porosity[1:,1:]) * F[:-1,:-1,2] * porosity_2
#     newF[:-1,:,7] = F[1:,:,7] * (1 - (1 - porosity[:-1,:]) * porosity_7) \
#                 + (1-porosity[1:,:]) * F[:-1,:,3] * porosity_3
#     newF[:-1,1:,8] = F[1:,:-1,8] * (1 - (1 - porosity[:-1,1:]) * porosity_8) \
#                 + (1-porosity[1:,:-1]) * F[:-1,1:,4] * porosity_4
    
#     ### corners
    
#     newF[1,0,2] = F[1,0,6]
#     newF[-2,0,8] = F[-2,0,4]
#     newF[1,-1,4] = F[1,-1,8]
#     newF[-2,-1,6] = F[-2,-1,2]
    
#     return newF

def DriftF(F,porosity):
    '''Pushes the F distribution along their respective axes with a partial 
    bounce-back against cells of smaller porosity than the incoming cell'''
    
    newF = np.zeros_like(F)
    
    # is porosity in each of the 8 directions smaller than inflowing cell ?
    
    porosity_1 = np.minimum(1, np.divide(porosity[:,1:], porosity[:,:-1], out=np.zeros_like(porosity[:,1:]), where=(porosity[:,:-1]!=0)))
    porosity_2 = np.minimum(1, np.divide(porosity[1:,1:], porosity[:-1,:-1], out=np.zeros_like(porosity[1:,1:]), where=(porosity[:-1,:-1]!=0)))
    porosity_3 = np.minimum(1, np.divide(porosity[1:,:], porosity[:-1,:], out=np.zeros_like(porosity[1:,:]), where=(porosity[:-1,:]!=0)))
    porosity_4 = np.minimum(1, np.divide(porosity[1:,:-1], porosity[:-1,1:], out=np.zeros_like(porosity[1:,:-1]), where=(porosity[:-1,1:]!=0)))
    porosity_5 = np.minimum(1, np.divide(porosity[:,:-1], porosity[:,1:], out=np.zeros_like(porosity[:,:-1]), where=(porosity[:,1:]!=0)))
    porosity_6 = np.minimum(1, np.divide(porosity[:-1,:-1], porosity[1:,1:], out=np.zeros_like(porosity[:-1,:-1]), where=(porosity[1:,1:]!=0)))
    porosity_7 = np.minimum(1, np.divide(porosity[:-1,:], porosity[1:,:], out=np.zeros_like(porosity[:-1,:]), where=(porosity[1:,:]!=0)))
    porosity_8 = np.minimum(1, np.divide(porosity[:-1,1:], porosity[1:,:-1], out=np.zeros_like(porosity[:-1,1:]), where=(porosity[1:,:-1]!=0)))
    
    
    newF[:,:,0] = F[:,:,0]
    
    newF[:,1:,1] = F[:,:-1,1] * porosity_1 \
                + F[:,1:,5] * (1 - porosity_5)
                
    newF[1:,1:,2] = F[:-1,:-1,2] * porosity_2 \
                + F[1:,1:,6] * (1 - porosity_6)
                
    newF[1:,:,3] = F[:-1,:,3] * porosity_3 \
                + F[1:,:,7] * (1 - porosity_7)
                
    newF[1:,:-1,4] = F[:-1,1:,4] * porosity_4 \
                + F[1:,:-1,8] * (1 - porosity_8)
                
    newF[:,:-1,5] = F[:,1:,5] * porosity_5 \
                + F[:,:-1,1] * (1 - porosity_1)
                
    newF[:-1,:-1,6] = F[1:,1:,6] * porosity_6 \
                + F[:-1,:-1,2] * (1 - porosity_2)
                
    newF[:-1,:,7] = F[1:,:,7] * porosity_7 \
                + F[:-1,:,3] * (1 - porosity_3)
                
    newF[:-1,1:,8] = F[1:,:-1,8] * porosity_8 \
                + F[:-1,1:,4] * (1 - porosity_4)
    
    ### corners
    
    newF[1,0,2] = F[1,0,6]
    newF[-2,0,8] = F[-2,0,4]
    newF[1,-1,4] = F[1,-1,8]
    newF[-2,-1,6] = F[-2,-1,2]
    
    return newF

def StandstillDriftF(F,porosity):
    '''Pushes the F distribution along their respective axes with only a 
    certain amount let into less porous targets, the remainder being placed
    into f0 for redistribution via collision'''
    
    newF = np.zeros_like(F)
    
    # is porosity in each of the 8 directions smaller than inflowing cell ?
    
    porosity_1 = np.minimum(1, np.divide(porosity[:,1:], porosity[:,:-1], out=np.zeros_like(porosity[:,1:]), where=(porosity[:,:-1]!=0)))
    porosity_2 = np.minimum(1, np.divide(porosity[1:,1:], porosity[:-1,:-1], out=np.zeros_like(porosity[1:,1:]), where=(porosity[:-1,:-1]!=0)))
    porosity_3 = np.minimum(1, np.divide(porosity[1:,:], porosity[:-1,:], out=np.zeros_like(porosity[1:,:]), where=(porosity[:-1,:]!=0)))
    porosity_4 = np.minimum(1, np.divide(porosity[1:,:-1], porosity[:-1,1:], out=np.zeros_like(porosity[1:,:-1]), where=(porosity[:-1,1:]!=0)))
    porosity_5 = np.minimum(1, np.divide(porosity[:,:-1], porosity[:,1:], out=np.zeros_like(porosity[:,:-1]), where=(porosity[:,1:]!=0)))
    porosity_6 = np.minimum(1, np.divide(porosity[:-1,:-1], porosity[1:,1:], out=np.zeros_like(porosity[:-1,:-1]), where=(porosity[1:,1:]!=0)))
    porosity_7 = np.minimum(1, np.divide(porosity[:-1,:], porosity[1:,:], out=np.zeros_like(porosity[:-1,:]), where=(porosity[1:,:]!=0)))
    porosity_8 = np.minimum(1, np.divide(porosity[:-1,1:], porosity[1:,:-1], out=np.zeros_like(porosity[:-1,1:]), where=(porosity[1:,:-1]!=0)))
    
    bounceback_1 = (porosity[:,1:] == 0)
    bounceback_2 = (porosity[1:,1:] == 0)
    bounceback_3 = (porosity[1:,:] == 0)
    bounceback_4 = (porosity[1:,:-1] == 0)
    bounceback_5 = (porosity[:,:-1] == 0)
    bounceback_6 = (porosity[:-1,:-1] == 0)
    bounceback_7 = (porosity[:-1,:] == 0)
    bounceback_8 = (porosity[:-1,1:] == 0)
    
    newF[:,:,0] = F[:,:,0]
    newF[:,:-1,0] += F[:,:-1,1] * (1-porosity_1-bounceback_1)
    newF[:-1,:-1,0] += F[:-1,:-1,2] * (1-porosity_2-bounceback_2)
    newF[:-1,:,0] += F[:-1,:,3] * (1-porosity_3-bounceback_3)
    newF[:-1,1:,0] += F[:-1,1:,4] * (1-porosity_4-bounceback_4)
    newF[:,1:,0] += F[:,1:,5] * (1-porosity_5-bounceback_5)
    newF[1:,1:,0] += F[1:,1:,6] * (1-porosity_6-bounceback_6)
    newF[1:,:,0] += F[1:,:,7] * (1-porosity_7-bounceback_7)
    newF[1:,:-1,0] += F[1:,:-1,8] * (1-porosity_8-bounceback_8)
    
    newF[:,1:,1] = F[:,:-1,1] * porosity_1 \
                    + F[:,1:,5] * bounceback_5
                
    newF[1:,1:,2] = F[:-1,:-1,2] * porosity_2 \
                    + F[1:,1:,6] * bounceback_6
                
    newF[1:,:,3] = F[:-1,:,3] * porosity_3 \
                    + F[1:,:,7] * bounceback_7
                
    newF[1:,:-1,4] = F[:-1,1:,4] * porosity_4 \
                    + F[1:,:-1,8] * bounceback_8
                
    newF[:,:-1,5] = F[:,1:,5] * porosity_5 \
                    + F[:,:-1,1] * bounceback_1
                
    newF[:-1,:-1,6] = F[1:,1:,6] * porosity_6 \
                    + F[:-1,:-1,2] * bounceback_2
                
    newF[:-1,:,7] = F[1:,:,7] * porosity_7 \
                    + F[:-1,:,3] * bounceback_3
                
    newF[:-1,1:,8] = F[1:,:-1,8] * porosity_8 \
                    + F[:-1,1:,4] * bounceback_4
    
    ### corners
    
    newF[1,0,2] = F[1,0,6]
    newF[-2,0,8] = F[-2,0,4]
    newF[1,-1,4] = F[1,-1,8]
    newF[-2,-1,6] = F[-2,-1,2]
    
    return newF

def GetZhangBoundaries(F, ρ, ρ0, dP, Cs):
    '''Determines the unknown F values at the inlet and outlet to maintain
    a fixed pressure difference equal to dP * channel length'''
    
    Nx = np.shape(F)[1]
    
    # inlet BC
    F[:,0,1] = F[:,-1,1] * (ρ0 + dP/Cs**2) / np.mean(ρ[:,-1])
    F[1:,0,2] = F[1:,-1,2] * (ρ0 + dP/Cs**2) / np.mean(ρ[:,-1])
    F[:-1,0,8] = F[:-1,-1,8] * (ρ0 + dP/Cs**2) / np.mean(ρ[:,-1])
    
    # outlet BC 
    F[:,-1,5] = F[:,0,5] * (ρ0 - dP*(Nx-1)/Cs**2) / np.mean(ρ[:,0])
    F[1:,-1,4] = F[1:,0,4] * (ρ0 - dP*(Nx-1)/Cs**2) / np.mean(ρ[:,0])
    F[:-1,-1,6] = F[:-1,0,6] * (ρ0 - dP*(Nx-1)/Cs**2) / np.mean(ρ[:,0])

    return F


def MoveParticles(particles, ux, uy, l, h):
    '''For animations: given particle positions and a velocity field, compute 
    next step positions'''
    
    newParticles = []
    for [x,y] in particles:
        u = ux[int(y), int(x)]
        v = uy[int(y), int(x)]
        
        
        if np.sqrt(u**2 + v**2) < 0.001:
            x = 1
            y = np.random.randint(0,h)
        else:
            x += u
            y += v
            
        if x > l-1:
            x = 1
            y = np.random.randint(0,h)
        elif x < 0:
            x = l - x
        if y > h:
            y = h - (y-h)
        if y < 0:
            y = -y
        
        newParticles.append([x,y])
    return newParticles     