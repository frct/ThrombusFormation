# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:18:43 2023

@author: vf926215
"""

import numpy as np


''' ###########################################################################

BINDING FUNCTIONS

###########################################################################'''


def GetBindingPoints(density, stickiness):
    '''Finds cells which are both empty and next to at least one sticky cell,
    with stickiness provided per requirements'''
    
    could_bind = (
        (density[1:-1,1:-1] == 0)
        &
        (
            (stickiness[:-2,1:-1]>0)
            |
            (stickiness[2:,1:-1]>0)
            |
            (stickiness[1:-1,:-2]>0)
            |
            (stickiness[1:-1,2:]>0)
            |
            (stickiness[:-2,:-2]>0)
            |
            (stickiness[:-2,2:]>0)
            |
            (stickiness[2:,:-2]>0)
            |
            (stickiness[2:,2:]>0)
            )
        )
    
    true_indices = np.argwhere(could_bind) + 1 # because could_bind is defined only between 1:-1 rows and 1:-1 columns we have to increment by 1 to get indices in original array
    
    return true_indices, could_bind

def GetBindingProbability(i, j, stickiness, ux=None, uy=None,  u_ref = 1e-5, flow_dependence=False):
    '''Given a position i,j and stickiness array, returns a probability of
    binding by combining stickiness of surrounding cells optionally modulated
    by flow'''
    
    if flow_dependence:
    
        normal_vectors = [
        [1,0],
        [1/np.sqrt(2), 1/np.sqrt(2)],
        [0,1],
        [-1/np.sqrt(2), 1/np.sqrt(2)],
        [-1,0],
        [-1/np.sqrt(2), -1/np.sqrt(2)],
        [0,-1],
        [1/np.sqrt(2), -1/np.sqrt(2)]]
    
        contributions = np.zeros(8)
    
        for c in range(8):
            projection_normalised = 1 / u_ref * (normal_vectors[c][0] * ux[i,j] + normal_vectors[c][1] * uy[i,j])
            contributions[c] = 1/4 * (1/2 - np.arctan(-projection_normalised)/np.pi) #!!! minus projection is new
            
    else:
        contributions = np.ones(8)
    
    # contribution of horizontally and vertically adjacent cells to probability
    # of not binding
    p_adj = (
            (1 - contributions[0] * stickiness[i,j-1])
            *
            (1 - contributions[2] * stickiness[i-1,j])
            *
            (1 - contributions[4] * stickiness[i,j+1])
            *
            (1 - contributions[6] * stickiness[i+1,j])
            )
    
    # contribution of diagonally adjacent cells weighted by square root of 2
    # for distance effect
    p_diag = (
            (1 - contributions[1] * stickiness[i-1,j-1] / np.sqrt(2))
            *
            (1 - contributions[3] * stickiness[i-1,j+1] / np.sqrt(2))
            *
            (1 - contributions[5] * stickiness[i+1,j+1] / np.sqrt(2))
            *
            (1 - contributions[7] * stickiness[i+1,j-1] / np.sqrt(2))
            )
           
    return 1 - p_adj * p_diag

def plateletFill(density, cRow, cCol, platelet_density, activation):
    # occupancy - integer numpy array of 0s (empty) and 1s (filled)
    # cRow - array row number of fill centre
    # cCol - array column number of fill centre
    # partial - logical value, do we fill if not enough space?
        
    state = (density[cRow - 1:cRow + 2, cCol - 1:cCol + 2] > 0)
    new_density = density.copy()
    
    filledCells = np.sum(state)
    if filledCells > 5:
        state[:, :] = np.ones((3, 3), dtype = int)
    else:
        state[1, 1] = 1
        
        # Shift the activation array in all eight directions
        up = np.roll(activation, -1, axis=0)
        down = np.roll(activation, 1, axis=0)
        left = np.roll(activation, -1, axis=1)
        right = np.roll(activation, 1, axis=1)
        up_left = np.roll(np.roll(activation, -1, axis=0), -1, axis=1)
        up_right = np.roll(np.roll(activation, -1, axis=0), 1, axis=1)
        down_left = np.roll(np.roll(activation, 1, axis=0), -1, axis=1)
        down_right = np.roll(np.roll(activation, 1, axis=0), 1, axis=1)
        
        # Add the shifted arrays together
        surround_activation = up + down + left + right + up_left + up_right + down_left + down_right
        
        surround_activation = surround_activation[cRow - 1:cRow + 2, cCol - 1:cCol + 2]
        # Generate random numbers and add them to the surround_activation array
        surround_activation += np.random.rand(3, 3) * 1e-10  # Small random numbers to break ties

        surround_activation[1,1] = 0
        
        # make sure not to build into vessel walls
        if cRow == 1:
            surround_activation[0,:] = 0
        if cRow == activation.shape[0] - 2:
            surround_activation[2,:] = 0
            
        masked_activation = np.ma.masked_where(state != 0, surround_activation)
        
        indices_sorted = np.argsort(-masked_activation, axis=None)
        
        highest_indices = np.unravel_index(indices_sorted[:2], state.shape)

        # Mark the corresponding cells in the state array as 1
        state[highest_indices] = 1
        
        
    new_density[cRow - 1:cRow + 2, cCol - 1:cCol + 2] = state * platelet_density
    
    # make sure update doesn't transform bits of wall into platelets
    new_density[0,:] = 1
    new_density[-1,:] = 1
    
    return new_density

def BindPlatelets(stickiness, density, platelet_density, ux=None, uy=None, u_ref=None, flow_dependence=False):
    ''' Given a chosen definition of stickiness, goes through the process of 
    picking a random binding point and randomly binding a platelet'''
    
    did_bind = 0
    new_density = density.copy()
    
    true_indices, _ = GetBindingPoints(density, stickiness)
    
    if len(true_indices) > 0:
        i,j = true_indices[np.random.randint(len(true_indices))]
        p_bind = GetBindingProbability(i, j, stickiness, ux, uy,  u_ref, flow_dependence)
        
        if np.random.rand() < p_bind:
            did_bind = 1
            new_density = plateletFill(density, i, j, platelet_density, stickiness)
            
    return new_density, did_bind
    
''' ###########################################################################

DETACHMENT FUNCTIONS

########################################################################### '''


def GetDetachmentPoints(density, activation):
    ''' Find thrombus points which are in contact with blood (i.e. not 
    completely surrounded by thrombus) and with activation smaller than 1 '''
    
    # n_neighbours = (
    #     np.where(density[1:-1,2:]>0,1,0)
    #     +
    #     np.where(density[2:,2:]>0,1,0)
    #     +
    #     np.where(density[2:,1:-1]>0,1,0)
    #     +
    #     np.where(density[2:,:-2]>0,1,0)
    #     +
    #     np.where(density[1:-1,:-2]>0,1,0)
    #     +
    #     np.where(density[:-2,:-2]>0,1,0)
    #     +
    #     np.where(density[:-2,1:-1]>0,1,0)
    #     +
    #     np.where(density[:-2,2:]>0,1,0))
    
    could_detach = (activation[1:-1,1:-1] < 1) & (density[1:-1,1:-1] > 0) #& (n_neighbours < 8)
    
    true_indices = np.argwhere(could_detach) + 1
    
    return true_indices, could_detach

def GetDetachProbability(i, j, activation, p_max, activation_max = 1, density = None, ux=None, uy=None,  u_ref = 1e-5, flow_dependence=False, ρ=None, ρ0 = 1.06):
    ''' Computes the probability of detachment for a given position with 
    optional effect of flow'''
    
    flow_effect = 1
    
    if flow_dependence:
        normal_vectors = [
        [1,0],
        [1/np.sqrt(2), 1/np.sqrt(2)],
        [0,1],
        [-1/np.sqrt(2), 1/np.sqrt(2)],
        [-1,0],
        [-1/np.sqrt(2), -1/np.sqrt(2)],
        [0,-1],
        [1/np.sqrt(2), -1/np.sqrt(2)]]
        
        targets = [
            [i,j+1],
            [i+1,j+1],
            [i+1,j],
            [i+1,j-1],
            [i,j-1],
            [i-1,j-1],
            [i-1,j],
            [i-1,j+1]
            ]
        
        sources = targets[4:] + targets[:4]
        
        contributions = np.zeros(len(normal_vectors))
        
        if ρ is None:
            ρ_cell = 0.7 * ρ0 #porosity of a shell cell times plasma density
        else:
            ρ_cell = ρ[i,j]
        
        for c, vec in enumerate(normal_vectors):
            normalised_projection = np.maximum(ρ_cell / (u_ref*ρ0) * (vec[0] * ux[i,j] + vec[1] * uy[i,j]),0)
            
            if density[targets[c][0], targets[c][1]] > 0:
                x = 0
            elif density[sources[c][0],sources[c][1]] > 0:
                x = 1
            else:
                x = 2
                
            contributions[c] = x * (1 - np.exp(-normalised_projection))
        
        flow_effect = np.sum(contributions) / 7

      
    p_detach = p_max * flow_effect * (activation_max - activation[i,j])
        
    return p_detach, flow_effect

def plateletRemove(density, cRow, cCol, platelet_density, activation, dice_roll):

    new_density = density.copy()
    
    state = (density[cRow - 1:cRow + 2, cCol - 1:cCol + 2] >= platelet_density)
    surrounding_activation = activation[cRow - 1:cRow + 2, cCol - 1:cCol + 2].copy()
    
    # we already know the [1,1] position is going, so exclude it from sorting process
    surrounding_activation[1,1] = np.inf 
    
    indices_below_threshold = np.where((surrounding_activation < dice_roll) & state)
    
    n_removed = 1 # at the very least, the central cell is emptied
    
    if len(indices_below_threshold[0]) < 3:
        state[indices_below_threshold] = 0
        n_removed += len(indices_below_threshold[0])
    else:
        
        # Sort the indices based on activation values
        sorted_indices = sorted(zip(*indices_below_threshold), key=lambda idx: surrounding_activation[idx])
    
        # Take at most two lowest activations
        indices_to_set_to_zero = sorted_indices[:2]
        n_removed += len(indices_to_set_to_zero)
        
        for idx in indices_to_set_to_zero:
            state[idx] = 0
    
    # remove central cell as well
    state[1,1] = 0
    new_density [cRow - 1:cRow + 2, cCol - 1:cCol + 2] = state * platelet_density
    
    # make sure update doesn't transform bits of wall into plasma
    new_density[0,:] = 1
    new_density[-1,:] = 1
    
    return new_density, n_removed/3

def DetachPlatelets(old_density, density, platelet_density, activation, p_max, activation_max = 1, ux=None, uy=None, u_ref=None, flow_dependence=False, constant_detachment = False, ρ=None):
    '''Picks a random platelet at the interface with blood stream and randomly
    removes it depending on its activation level and optionally flow'''
    
    did_detach = 0
    
    new_density = density.copy()
    
    # to avoid ordering effects, we use the old copy of density to identify
    # detachment points
    true_indices, _ = GetDetachmentPoints(old_density, activation)
    
    if len(true_indices) > 0:
        i,j = true_indices[np.random.randint(len(true_indices))]
        if constant_detachment:
            p_detach = p_max
        else:
            p_detach,_ = GetDetachProbability(i, j, activation, p_max, activation_max, old_density, ux, uy, u_ref, flow_dependence, ρ=ρ, ρ0 = 1.06)
        
        dice_roll = np.random.rand()
        
        if dice_roll < p_detach:
            # when actually removing, use the density post attachment array
            new_density, did_detach = plateletRemove(density, i, j, platelet_density, activation, dice_roll)
            
    return new_density, did_detach

def GetNeighbours(cell,Nx=256,Ny=64):
    y,x = cell
    adj = []
    diag = []
    if x > 0:
        adj.append([y,x-1])
        if y < Ny-2:
            adj.append([y+1,x])
            diag.append([y+1,x-1])
        if y > 1:
            adj.append([y-1,x])
            diag.append([y-1,x-1])
            
    if x < Nx-1:
        adj.append([y,x+1])
        if y < Ny-2:
            adj.append([y+1,x])
            diag.append([y+1,x+1])
        if y > 1:
            adj.append([y-1,x])
            diag.append([y-1,x+1])

    return adj, diag
     
def RemoveUntethered(density, inj_start, inj_end):
    Ny, Nx = np.shape(density)
    X,Y = np.meshgrid(np.arange(Nx),np.arange(Ny))
    test = (X>= inj_start) & (X<inj_end) & (Y==1)
    anchored = np.zeros_like(test)
    
    update = True
    n_removed = 0
    
    while update:
        update = False
        idxs,idys = np.where(test)
        
        for cell in zip(idxs,idys):
            
            # remove cell from seed to avoid repeatedly testing and include in 
            # old seed to mark it as already tested
            
            test[cell[0],cell[1]] = 0
            anchored[cell[0],cell[1]] = 1
            
            # get list of neighbours and test if each one contains stuff and has not been previously seen
            adj, diag = GetNeighbours(cell, Nx=Nx, Ny=Ny)
            
            for neighbour in adj+diag:
                i,j = neighbour
                if (density[i,j] > 0) and not (test[i,j]==1 or anchored[i,j]==1) : #and (not (test[i,j]==1 || anchored[i,j]==1)):
                    if i>0:
                        test[i,j] = 1
                        update = True
                    
    # now empty any remaining cells which are not anchored
    untethered = (density[1:-1,:]>0) & (anchored[1:-1,:]==0)
    
    if np.any(untethered):
        density[1:-1,:][untethered] = 0
        n_removed = np.sum(untethered) / 3 # counting the number of cells we remove and dividing by 3, the standard platelet size to get number of detached platelets
    
    return density, n_removed

''' ##########################################################################

ACTIVATION FUNCTIONS

########################################################################### '''


def InstantaneousActivation(activation_state, density, MAX_ACTIVATION=1, loss=1):
    
    neighbouring_states = np.array([activation_state[:-2,1:-1:],
                                    activation_state[2:,1:-1],
                                    activation_state[1:-1,:-2],
                                    activation_state[1:-1,2:],
                                    activation_state[:-2,:-2],
                                    activation_state[:-2,2:],
                                    activation_state[2:,:-2],
                                    activation_state[2:,2:]])
    
    neighbouring_occupancies = np.array([density[:-2,1:-1:],
                                         density[2:,1:-1],
                                         density[1:-1,:-2],
                                         density[1:-1,2:],
                                         density[:-2,:-2],
                                         density[:-2,2:],
                                         density[2:,:-2],
                                         density[2:,2:]])
    
    Σ_signals = np.sum(neighbouring_states, axis = 0)
    N_neighbours = np.sum(neighbouring_occupancies > 0, axis = 0)
    f = Σ_signals / (N_neighbours + loss * (8 - N_neighbours))
    
    final_activation = np.maximum(activation_state[1:-1, 1:-1], f)

    mask = (activation_state[1:-1,1:-1] < final_activation) & (density[1:-1,1:-1]>0)
    activation_state[1:-1,1:-1][mask] = final_activation[mask]
    activation_state[activation_state>MAX_ACTIVATION] = MAX_ACTIVATION
    activation_state[density == 0] = 0
    
    return activation_state

def SigmoidActivation(activation_state, density, MAX_ACTIVATION, Δt, k, ϵ = 0.001, loss = 1):

    neighbouring_states = np.array([activation_state[:-2,1:-1:],
                                    activation_state[2:,1:-1],
                                    activation_state[1:-1,:-2],
                                    activation_state[1:-1,2:],
                                    activation_state[:-2,:-2],
                                    activation_state[:-2,2:],
                                    activation_state[2:,:-2],
                                    activation_state[2:,2:]])
    
    neighbouring_occupancies = np.array([density[:-2,1:-1:],
                                         density[2:,1:-1],
                                         density[1:-1,:-2],
                                         density[1:-1,2:],
                                         density[:-2,:-2],
                                         density[:-2,2:],
                                         density[2:,:-2],
                                         density[2:,2:]])
    
    Σ_signals = np.sum(neighbouring_states, axis = 0)
    N_neighbours = np.sum(neighbouring_occupancies > 0, axis = 0)
    f = Σ_signals / (N_neighbours + loss * (8 - N_neighbours))
    
    final_activation = np.maximum(activation_state[1:-1, 1:-1], f) #α * np.mean(neighbouring_states**power,axis=0))
    normalised_activation = activation_state[1:-1,1:-1].copy()
    
    # to avoid division by 0 when normalising
    zero_mask = (final_activation > 0)
    normalised_activation[zero_mask] /= final_activation[zero_mask]
    
    # find newly attached platelets, and initialise activation to non zero
    new_platelets_mask = (final_activation>0) & (normalised_activation == 0) & (density[1:-1,1:-1] > 0)
    normalised_activation[new_platelets_mask] = ϵ * final_activation[new_platelets_mask]
    
    dactivation = final_activation * k * normalised_activation * (1-normalised_activation) * Δt
    dactivation[dactivation<0] = 0
    activation_state[1:-1,1:-1] += dactivation
    mask_overshoot = (activation_state[1:-1,1:-1] > final_activation)
    activation_state[1:-1,1:-1][mask_overshoot] = final_activation[mask_overshoot]
    activation_state[density == 0] = 0
    activation_state[activation_state>MAX_ACTIVATION] = MAX_ACTIVATION
    
    if np.isnan(activation_state).any():
        print('error')
        pass
    
    return activation_state  

def FixedActivation(activation_state, density, MAX_ACTIVATION, final_activation, Δt, k, ϵ = 0.001, loss = 1):
    ''' Instead of determining a final activation level based on surrounding 
    activation levels, we set a constant final activation which all thrombus 
    cells converge to'''
    
    
    #final_activation = np.maximum(activation_state[1:-1, 1:-1], MAX_ACTIVATION), #α * np.mean(neighbouring_states**power,axis=0))
    normalised_activation = activation_state[1:-1,1:-1] / final_activation

    
    # find newly attached platelets, and initialise activation to non zero
    new_platelets_mask = (normalised_activation == 0) & (density[1:-1,1:-1] > 0)
    normalised_activation[new_platelets_mask] = ϵ * final_activation
    
    dactivation = final_activation * k * normalised_activation * (1-normalised_activation) * Δt
    dactivation[dactivation<0] = 0
    activation_state[1:-1,1:-1] += dactivation
    # mask_overshoot = (activation_state[1:-1,1:-1] > final_activation)
    # activation_state[1:-1,1:-1][mask_overshoot] = final_activation
    activation_state[density == 0] = 0
    activation_state[activation_state>MAX_ACTIVATION] = MAX_ACTIVATION
    
    if np.isnan(activation_state).any():
        print('error')
        pass
    
    return activation_state  
    

##############################################################################

def GetThrombusHeight(thrombus):
    ''' Finds highest row number containing a piece of thrombus '''
    
    height = 0
    for row in range(1,np.shape(thrombus)[0]):
        if np.any(thrombus[row,:]):
            height += 1
        else:
            break
    return height
    