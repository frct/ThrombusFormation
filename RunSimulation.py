# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:36:39 2024

@author: vf926215
"""


import numpy as np
from tqdm import tqdm
from PlateletModel import BindPlatelets, DetachPlatelets, RemoveUntethered, \
    SigmoidActivation, InstantaneousActivation, FixedActivation, \
    GetBindingProbability, GetDetachProbability, NewDriftPlatelets
from LBM_functions import InitialiseLBM, UpdateLBM
import pickle
from scipy.optimize import fsolve
from scipy.stats import beta

'''############################################################################
HELPER FUNCTIONS TO CALIBRATE DETACHMENT RATE
############################################################################'''

# def GetExpectedDetachmentTime(p, Δt):
#     N_timesteps = int(10 / Δt)
#     p_onepiece = np.zeros(N_timesteps)
#     p_twopiece = np.zeros(N_timesteps)
#     for t in range(1,N_timesteps):
#         p_onepiece[t] = (1-p)**(t-1) * p/3
#         if t>1:
#             p_twopiece[t] = (t-1) * 2/3 * p**2 * (1-p) ** (t-2)
#     p_detach = p_onepiece + p_twopiece
#     ExpectedWait = np.sum(np.arange(N_timesteps) * p_detach) * Δt
    
#     return ExpectedWait

def GetExpectedDetachmentTime(p, Δt):
    N_timesteps = int(10 / Δt)
    t = np.arange(1, N_timesteps)  # start from 1

    # Vectorized one-piece and two-piece probabilities
    p_onepiece = (1 - p) ** (t - 1) * p / 3
    p_twopiece = np.zeros_like(p_onepiece)
    p_twopiece[1:] = (t[1:] - 1) * (2/3) * p**2 * (1 - p) ** (t[1:] - 2)

    p_detach = p_onepiece + p_twopiece
    ExpectedWait = np.sum(t * p_detach) * Δt

    return ExpectedWait

def GetPDetach(DETACHMENT_TIME_SEC, Δt, run_test=False):
    
    try:
        [time, p] = pickle.load(open('pd=f(detachment_time).pkl', 'rb'))
    except:
        test_values = np.linspace(1e-9, 1e-6, 1000)               
        Expected_wait = np.zeros(len(test_values))        
        for i,p in enumerate(test_values):
            Expected_wait[i] = GetExpectedDetachmentTime(p, Δt)
        time = Expected_wait[::-1]
        p = test_values[::-1]           
        pickle.dump([time,p], open('pd=f(detachment_time).pkl', 'wb'))

    if DETACHMENT_TIME_SEC > time[-1] or DETACHMENT_TIME_SEC < time[0]:
        raise Exception('Detachment time not appropriate for interpolation')
    else:
        desired_p = np.interp(DETACHMENT_TIME_SEC,time,p)
        
    if run_test:
        test = GetExpectedDetachmentTime(desired_p, Δt)
        print(f'P_detachment = {desired_p} gives an expected detachment time = {test}')

    return desired_p

def GetFlowDpdntPDetach(DETACHMENT_TIME_SEC, Δt, density, activation, u_ref=None):
    
    Ny, Nx = np.shape(density)
    
    density[2,int(Nx/2-1):int(Nx/2+2)] = 0.3
    porosity = 1 - density

    try:
        F, ux, uy, vel = pickle.load(open('flow detachment test flow.pkl', 'rb'))
    except:
        Δx_USI = 1e-6 # lattice unit (lu) size in m
        C_ρ = 1000 # conversion factor for ρ in kg/m3
        ρ_USI = 1060 # kg/m3
        HEIGHT_USI = (Ny-2) * Δx_USI # m
        RADIUS_USI = HEIGHT_USI / 2 # m
        γ_USI = 1000 # s-1
        U_MAX_USI = γ_USI * RADIUS_USI / 2 # m/s
        μ_USI = 4e-3 # extrapolation from Cherry 2013 and lab calculation for in vitro experiments
        NU_USI = μ_USI / ρ_USI
        τ = 0.809 # dimensionless characteristic relaxation time, must be bigger than 0.5, ideal value = 0.809
        
        # INITIALISE LBM
        
        CELERITY_OF_SOUND_LBM, U_MAX_LBM, NU_LBM, Δt_LBM, ρ0, dP_dx, F, ρ, ux_old, uy_old = InitialiseLBM(Nx, Ny, Δx_USI, τ, NU_USI, ρ_USI, U_MAX_USI, C_ρ)
        
        
        F = np.einsum('ijk,ij->ijk', F, porosity)
        F, ux, uy, vel, *_ = UpdateLBM(porosity, F, ρ0, τ, dP_dx, CELERITY_OF_SOUND_LBM, N_convergence=100, is_print=False)      
        pickle.dump([F, ux, uy, vel], open('flow detachment test flow.pkl', 'wb'))

    p_max = 0.017 #doesn't matter but necessary to run GetDetachProbability and get flow_effect
    if u_ref is None:
        u_ref = vel[2,128] 
    _, flow_effect = GetDetachProbability(2, 128, activation, p_max, density = density, ux=ux, uy=uy, u_ref = u_ref, flow_dependence=True)

    desired_rate = Δt / DETACHMENT_TIME_SEC
    desired_p = desired_rate / flow_effect    
    return desired_p

'''############################################################################
MAIN FUNCTION
############################################################################'''


def RunSimulation(
        save_file=None,
        constant_binding=False,
        constant_detachment=False,
        final_activation = None,
        flow_dependence=False,
        want_flow=False,
        want_frames = False,
        want_trajectory = False,
        want_core = False,
        Nx = 256,
        Ny = 64,
        INJURY_LENGTH = 50,
        T = 60,
        PLATELET_COUNT = 200000, # in platelets/microL
        BETA = 0.01,
        MAX_ACTIVATION = 1,
        EPSILON_ACTIVATION = 0.001,
        HALF_ACTIVATION_SEC = 1,
        ACTIVATION_LOSS = 1,
        DETACHMENT_TIME_SEC = np.inf,
        u_ref_bind = None,
        u_ref_detach = 2e-4,
        PLATELET_DENSITY = 0.3,
        core_threshold = 0.7,
        core_density = 0.7,
        CFL = 0.8,
        fps = 15,
        gif_duration = 10):

###############################################################################
    # SIMULATION SETUP

        
    INJURY_START = (Nx - INJURY_LENGTH) // 2
    INJURY_END = INJURY_START + INJURY_LENGTH
    
    density = np.zeros((Ny,Nx))    
    density[0,:] = 1 # vessel walls
    density[-1,:] = 1
    if want_core and core_threshold<1:
        density[1,INJURY_START:INJURY_END] = core_density
    else:
        density[1,INJURY_START:INJURY_END] = PLATELET_DENSITY
    porosity = 1 - density
    
    activation = np.zeros((Ny,Nx))
    activation[1, INJURY_START:INJURY_END] = MAX_ACTIVATION
            
###############################################################################
    # LBM SETUP
    
    
    # LBM PARAMETERS
    
    Δx_USI = 1e-6 # lattice unit (lu) size in m
    C_ρ = 1000 # conversion factor for ρ in kg/m3
    ρ_USI = 1060 # kg/m3
    LENGTH_USI = Nx * Δx_USI # m
    HEIGHT_USI = (Ny-2) * Δx_USI # m
    RADIUS_USI = HEIGHT_USI / 2 # m
    γ_USI = 1000 # s-1
    U_MAX_USI = γ_USI * RADIUS_USI / 2 # m/s
    μ_USI = 4e-3 # extrapolation from Cherry 2013 and lab calculation for in vitro experiments
    NU_USI = μ_USI / ρ_USI
    τ = 0.809 # dimensionless characteristic relaxation time, must be bigger than 0.5, ideal value = 0.809

    # INITIALISE LBM
    
    CELERITY_OF_SOUND_LBM, U_MAX_LBM, NU_LBM, Δt_LBM, ρ0, dP_dx, F, ρ, ux, uy = InitialiseLBM(Nx, Ny, Δx_USI, τ, NU_USI, ρ_USI, U_MAX_USI, C_ρ)
    vel = np.sqrt(ux**2 + uy**2)
    if want_core:
        initial_save_name = 'Initial flow over core.pkl'
    else:
        initial_save_name = 'Initial flow.pkl'
    
    try:
        print('Loading previous save of first frame')
        F, ux, uy, vel, ρ = pickle.load(open(initial_save_name, 'rb'))
    except:
        print('Loading failed, calculating from scratch')
        F = np.einsum('ijk,ij->ijk', F, porosity)
        F, ux, uy, vel, ρ, _ = UpdateLBM(porosity, F, ρ0, τ, dP_dx, CELERITY_OF_SOUND_LBM, N_convergence=100, is_print=False)      
        pickle.dump([F, ux, uy, vel, ρ], open(initial_save_name, 'wb'))
        

##############################################################################    
    # EXTRACT DEPENDENT VARIABLES
    
    PLATELET_COUNT_USI = PLATELET_COUNT * 10**9 # e.g. 200,000 plts/microL => 200.10^12 plts/m3
    N_PLATELETS = int(PLATELET_COUNT_USI * np.pi * RADIUS_USI**2 * LENGTH_USI) # N = Concentration x Volume
    PLATELET_RATIO = N_PLATELETS / (Nx * (Ny-2) - INJURY_LENGTH) # proportion of cells occupied by a platelet
    
    
    
    # INITIALISE PLATELET POSITIONS
    
    shape_param = 0.34
    Y = 1 + 1e-10 + beta.rvs(a=shape_param, b=shape_param, size=N_PLATELETS) * (Ny-2-2e-10)
    X = np.random.uniform(0, Nx, N_PLATELETS)        
    platelets = [[x, y] for x,y in zip(X,Y)]
 
                
    
    # kB = 1.38e-23
    # Temp = 310
    # R = 1e-6 # 10e-12
    # D = kB * Temp / (6 * np.pi * μ_USI * R)
    
    Δt = CFL / np.max(np.sqrt(ux**2 + uy**2)) * Δt_LBM # timestep in s
    Nt = int(T / Δt) + 1
    
    σ_diffusion = 0.05 #np.sqrt(2 * D * Δt / Δx_USI**2)
   
    
    if DETACHMENT_TIME_SEC == np.inf:
        P_DETACH_MAX = 0
        ρ = None
    elif flow_dependence:
        P_DETACH_MAX = GetFlowDpdntPDetach(DETACHMENT_TIME_SEC, Δt, density, activation, u_ref=u_ref_detach)
    else:
        P_DETACH_MAX = GetPDetach(DETACHMENT_TIME_SEC, Δt)
        ρ = None
        
    
    if HALF_ACTIVATION_SEC > 0:
        ACTIVATION_RATE = np.log((1 - EPSILON_ACTIVATION) / EPSILON_ACTIVATION) / HALF_ACTIVATION_SEC
        # ACTIVATION_RATE does not need to be calibrated w.r.t Δt as Δt and HALF_ACTIVATION_SEC are in the same unit
    else:
        ACTIVATION_RATE = None

###############################################################################
    # INITIALISE FRAMES
    
    if want_frames:
        N_frames = int(gif_duration * fps) + 1
        
        if N_frames > Nt:
            N_frames = Nt
            gif_duration = Nt / fps
        
        Δt_frame = T / (N_frames-1)
        frame_times = iter([Δt_frame * i for i in range(N_frames+1)])
        
        save_time = next(frame_times)
        frame = 0
        
        
###############################################################################        
    # INITIALISE RECORDING VARIABLES
    
    clot_size = np.zeros((Nt))
    binding_events = np.zeros((Nt))
    detachment_events = np.zeros((Nt))

    if want_core:
        core_size = np.zeros((Nt))
        
    platelet_count = np.zeros((Nt))
    
    if want_trajectory:
        trajectory_save_interval = 0.001
        n_save = 0
        next_trajectory_save_time = n_save * trajectory_save_interval
    
    last_save = 0
    
    for t in tqdm(range(Nt)):
        
        current_time = t * Δt
        
        if want_trajectory and current_time >= next_trajectory_save_time:            
             pickle.dump([platelets, density, activation, current_time], open(f'trajectories/platelet positions {int(n_save)}.pkl', 'wb'))
             n_save += 1
             next_trajectory_save_time = n_save * trajectory_save_interval

        
        # SNAPSHOT OF CURRENT STATE
        
        clot_size[t] = np.sum(density[1:-1,:]>0) - INJURY_LENGTH
        
        platelets = NewDriftPlatelets(platelets, ux, uy, density, Δt/Δt_LBM, σ_diffusion)
        
        platelet_count[t] = len(platelets)
        ratio = platelet_count[t] / (Nx * (Ny-2) - INJURY_LENGTH - clot_size[t])
        
        if ratio < PLATELET_RATIO:
            # use rejection sampling to sample from distribution proportional
            # to beta * ux
            Y = None
            while Y is None:
                proposed_Y = 1 + 1e-10 + beta.rvs(a=shape_param, b=shape_param) * (Ny-2-2e-10)
                v = ux[int(proposed_Y),0]
                accept_prob = v / np.max(ux[1:-1, 0])
                if np.random.rand() < accept_prob:
                    Y = proposed_Y
                
            platelets.append([0, Y])
   
        
        
        if want_core:
            core_size[t] = np.sum(density[1:-1,:]==core_density) - INJURY_LENGTH
        
        if want_frames and current_time >= save_time:
            to_save = {'density': density,
                       'activation': activation,
                       'time': current_time,
                       'fps': fps,
                       'gif duration': gif_duration,
                       'frame interval': Δt_frame,
                       'clot size': clot_size[last_save:t+1],
                       'binding_events': binding_events[last_save:t+1],
                       'detachment_events': detachment_events[last_save:t+1],
                       'platelet count': platelet_count[last_save:t+1]
                       }
            if want_flow:
                to_save['ux'] = ux
                to_save['uy'] = uy
                to_save['rho'] = np.sum(F, axis=2)
            
            pickle.dump(to_save, open(f'frames/frame {frame}.pkl', 'wb'))
            last_save = t
            save_time = next(frame_times)
            frame += 1
        
        
        # get the stickiness depending on binding rule
        
        thrombus = np.zeros_like(density)
        thrombus[1:-1,:] = (density[1:-1] > 0).astype(int)
        
        if constant_binding:
            stickiness = BETA * thrombus
        else:
            stickiness = BETA * thrombus * activation
            
        # RUN UPDATES
        
        ''' BindPlatelets and DetachPlatelets use the same former copy of density 
        to avoid sequence effects'''
        
        density_post_attachment, binding_events[t], platelets = BindPlatelets(stickiness, density,  platelets, PLATELET_DENSITY, ux, uy, u_ref_bind)        
        density_post_detachment, detachment_events[t] = DetachPlatelets(density, density_post_attachment, PLATELET_DENSITY, activation, P_DETACH_MAX, MAX_ACTIVATION, ux, uy, u_ref_detach, flow_dependence, constant_detachment, ρ)
        new_density, n_removed = RemoveUntethered(density_post_detachment, INJURY_START, INJURY_END)    
        detachment_events[t] += n_removed
        
        # remove platelets that have become stuck in the thrombus
        platelets = [p for p in platelets if density[int(p[1]), int(p[0])] == 0]
        
        if final_activation is None: # in this case, the final activation depends on surrounding activation
            if ACTIVATION_RATE is None: # activation jumps instantaneously to successive final activation levels
                activation = InstantaneousActivation(activation, new_density, MAX_ACTIVATION, ACTIVATION_LOSS)
            else:
                activation = SigmoidActivation(activation, new_density, MAX_ACTIVATION, Δt, ACTIVATION_RATE, EPSILON_ACTIVATION, ACTIVATION_LOSS)
        else:
            activation = FixedActivation(activation, new_density, MAX_ACTIVATION, final_activation, Δt, ACTIVATION_RATE, EPSILON_ACTIVATION, ACTIVATION_LOSS)
        
        # if np.any(activation[1,INJURY_START:INJURY_END] != 1):
        #     break
        
        if want_core:
            new_density[activation>core_threshold] = core_density

        # launch LBM iff flow dependence and there is a change to the thrombus structure
        
        if flow_dependence and np.any(density != new_density):
            new_porosity = 1 - new_density
            porosity_ratio = np.nan_to_num(new_porosity / porosity, 1)
            F = np.einsum('ijk,ij->ijk', F, porosity_ratio)
            F, ux, uy, vel, ρ, *_ = UpdateLBM(new_porosity, F, ρ0, τ, dP_dx, CELERITY_OF_SOUND_LBM, N_convergence=10, is_print=False)
            porosity = new_porosity.copy()
    
        density = new_density.copy()
        
        if np.any(density[0,:]) < 1:
            raise Exception('Piece of vessel wall is gone')
    
###############################################################################
    # SAVE SIMULATION PARAMETERS AND RESULTS    
    
    save_content = {
    'constant_binding' : constant_binding,
    'constant_detachment' : constant_detachment,
    'flow_dependence': flow_dependence,
    'T': T,
    'Δt': Δt,
    'DETACHMENT_TIME_SEC': DETACHMENT_TIME_SEC,
    'MAX_ACTIVATION': MAX_ACTIVATION,
    'EPSILON_ACTIVATION': EPSILON_ACTIVATION,
    'HALF_ACTIVATION_SEC': HALF_ACTIVATION_SEC,
    'ACTIVATION_LOSS': ACTIVATION_LOSS,
    'P_DETACH_MAX': P_DETACH_MAX,
    'PLATELET_DENSITY': PLATELET_DENSITY,
    'clot size': clot_size,
    'binding_events': binding_events,
    'detachment_events': detachment_events,
    'final density': density,
    'final activation': activation,
    'environmental activation': final_activation is None,
    'free platelets': platelets,
    'platelet count': platelet_count
    }
    
    if want_core:
        save_content['core threshold'] = core_threshold
        save_content['core size'] = core_size
        save_content['core density'] = core_density
    if flow_dependence:
        save_content['u_ref_bind'] = u_ref_bind
        save_content['u_ref_detach'] = u_ref_detach
        save_content['final ux'] = ux
        save_content['final uy'] = uy
        save_content['rho'] = np.sum(F,axis=2)
    
    
    
    if save_file is not None:
        pickle.dump(save_content, open(save_file, 'wb'))
    
    return save_content
    
if __name__ == '__main__':
    res = RunSimulation(T=10, BETA=0.001, DETACHMENT_TIME_SEC=1, want_frames=False, want_flow=True)