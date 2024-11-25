# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:34:32 2024

Extract simulations data from pkl to export to excel spreadsheet

@author: vf926215
"""

import pickle
import pandas as pd

'''############################################################################
CACD SIMULATIONS
############################################################################'''

N_SIMULATIONS = 10
EXP_DET_TIMES_SEC = [1, 0.1, 0.05]

clot_sizes = {f't={t_det}': [] for t_det in EXP_DET_TIMES_SEC}

for i, t_det in enumerate(EXP_DET_TIMES_SEC):
    
    for simulation_nb in range(1,N_SIMULATIONS+1):
        save_name = f'CACD/Simulation {simulation_nb} with td = {t_det}.pkl'
        results = pickle.load(open(save_name, 'rb'))
        clot_sizes[f't={t_det}'].append(results['clot size'])


with pd.ExcelWriter('CACD.xlsx', engine="xlsxwriter") as writer:
    for t_det, data in clot_sizes.items():
        # Convert the data dictionary to a DataFrame for the current condition
        df = pd.DataFrame(data).T
        Δt = results['Δt']
        df.index = [f'{Δt * i} s' for i in range(len(results['clot size']))]
        df.columns = [f'simulation {i}' for i in range(1, N_SIMULATIONS+1)]

        df.to_excel(writer, sheet_name=f't_det = {t_det}')    
  
'''############################################################################
FIXED ACTIVATION MODEL WITH DIFFERENT DETACHMENT RATES
############################################################################'''

N_SIMULATIONS = 10
EXP_DET_TIMES_SEC = [0.05, 0.03, 0.02]

clot_sizes = {f't={t_det}': [] for t_det in EXP_DET_TIMES_SEC}

for i, t_det in enumerate(EXP_DET_TIMES_SEC):
    
    for simulation_nb in range(1,N_SIMULATIONS+1):
        save_name = f'FA/test detachment rate/Simulation {simulation_nb} with E(Δt_det)={t_det}.pkl'
        results = pickle.load(open(save_name, 'rb'))
        clot_sizes[f't={t_det}'].append(results['clot size'])


with pd.ExcelWriter('FA det rates.xlsx', engine="xlsxwriter") as writer:
    for t_det, data in clot_sizes.items():
        # Convert the data dictionary to a DataFrame for the current condition
        df = pd.DataFrame(data).T
        Δt = results['Δt']
        df.index = [f'{Δt * i} s' for i in range(len(results['clot size']))]
        df.columns = [f'simulation {i}' for i in range(1, N_SIMULATIONS+1)]

        df.to_excel(writer, sheet_name=f't_det = {t_det}')    

'''############################################################################
FA different activation rates
############################################################################'''

N_SIMULATIONS = 10
HALF_ACTIVATION_TIMES_SEC = [1, 5, 10]

clot_sizes = {f't={t50}': [] for t50 in HALF_ACTIVATION_TIMES_SEC}

for i, t50 in enumerate(HALF_ACTIVATION_TIMES_SEC):
   
    for simulation_nb in range(1,N_SIMULATIONS+1):
        save_name = f'FA/test t50/Simulation {simulation_nb} with t50 = {t50}.pkl'
        results = pickle.load(open(save_name, 'rb'))
        clot_sizes[f't={t50}'].append(results['clot size'])


with pd.ExcelWriter('FA act rates.xlsx', engine="xlsxwriter") as writer:
    for t_det, data in clot_sizes.items():
        # Convert the data dictionary to a DataFrame for the current condition
        df = pd.DataFrame(data).T
        Δt = results['Δt']
        df.index = [f'{Δt * i} s' for i in range(len(results['clot size']))]
        df.columns = [f'simulation {i}' for i in range(1, N_SIMULATIONS+1)]

        df.to_excel(writer, sheet_name=f't50 = {t50}')    

'''############################################################################
CAED simulations
############################################################################'''

N_SIMULATIONS = 10
EXP_DET_TIMES_SEC = [1, 0.1, 0.03]

clot_sizes = {f't={t_det}': [] for t_det in EXP_DET_TIMES_SEC}

for i, t_det in enumerate(EXP_DET_TIMES_SEC):
    
    for simulation_nb in range(1,N_SIMULATIONS+1):
        save_name = f'CAED/test detachment rate/Simulation {simulation_nb} of full model with td = {t_det}.pkl'
        results = pickle.load(open(save_name, 'rb'))
        clot_sizes[f't={t_det}'].append(results['clot size'])


with pd.ExcelWriter('CAED.xlsx', engine="xlsxwriter") as writer:
    for t_det, data in clot_sizes.items():
        # Convert the data dictionary to a DataFrame for the current condition
        df = pd.DataFrame(data).T
        Δt = results['Δt']
        df.index = [f'{Δt * i} s' for i in range(len(results['clot size']))]
        df.columns = [f'simulation {i}' for i in range(1, N_SIMULATIONS+1)]

        df.to_excel(writer, sheet_name=f't_det = {t_det}')    

'''############################################################################
EAnoD simulations
############################################################################'''

N_SIMULATIONS = 10
EXP_BINDING_TIMES_SEC = [0.05, 0.2, 0.5]

clot_sizes = {f't={t_bind}': [] for t_bind in EXP_BINDING_TIMES_SEC}

for i, t_bind in enumerate(EXP_BINDING_TIMES_SEC):
    for simulation_nb in range(1, N_SIMULATIONS+1):
        save_name = f'EAnoD/Simulation {simulation_nb} of no detachment model with initial binding time = {t_bind} for T =600'
        results = pickle.load(open(save_name, 'rb'))
        clot_sizes[f't={t_bind}'].append(results['clot size'])


with pd.ExcelWriter('EAnoD.xlsx', engine="xlsxwriter") as writer:
    for t_det, data in clot_sizes.items():
        # Convert the data dictionary to a DataFrame for the current condition
        df = pd.DataFrame(data).T
        Δt = results['Δt']
        df.index = [f'{Δt * i} s' for i in range(len(results['clot size']))]
        df.columns = [f'simulation {i}' for i in range(1, N_SIMULATIONS+1)]

        df.to_excel(writer, sheet_name=f't_det = {t_det}')
        
'''############################################################################
EAED with different detachment rates
############################################################################'''

N_SIMULATIONS = 10
EXP_DET_TIMES_SEC = [0.05, 0.1, 0.2]

clot_sizes = {f't={t_det}': [] for t_det in EXP_DET_TIMES_SEC}

for i, t_det in enumerate(EXP_DET_TIMES_SEC):
    
    for simulation_nb in range(1,N_SIMULATIONS+1):
        save_name = f'EAED/test detachment rate/Simulation {simulation_nb} of full model with td = {t_det}.pkl'
        results = pickle.load(open(save_name, 'rb'))
        clot_sizes[f't={t_det}'].append(results['clot size'])


with pd.ExcelWriter('EAED different detachment rates.xlsx', engine="xlsxwriter") as writer:
    for t_det, data in clot_sizes.items():
        # Convert the data dictionary to a DataFrame for the current condition
        df = pd.DataFrame(data).T
        Δt = results['Δt']
        df.index = [f'{Δt * i} s' for i in range(len(results['clot size']))]
        df.columns = [f'simulation {i}' for i in range(1, N_SIMULATIONS+1)]

        df.to_excel(writer, sheet_name=f't_det = {t_det}')
        
'''############################################################################
EAED with different activation rates
############################################################################'''

N_SIMULATIONS = 10
HALF_ACTIVATION_TIMES_SEC = [0, 0.2, 0.5, 1]

clot_sizes = {f't50={t50}': [] for t50 in HALF_ACTIVATION_TIMES_SEC}

for i, t50 in enumerate(HALF_ACTIVATION_TIMES_SEC):
   
    for simulation_nb in range(1,N_SIMULATIONS+1):
         save_name = f'EAED/test t50/Simulation {simulation_nb} of full model with t50 = {t50}.pkl'
         results = pickle.load(open(save_name, 'rb'))
         clot_sizes[f't50={t50}'].append(results['clot size'])


with pd.ExcelWriter('EAED different activation rates.xlsx', engine="xlsxwriter") as writer:
    for t50, data in clot_sizes.items():
        # Convert the data dictionary to a DataFrame for the current condition
        df = pd.DataFrame(data).T
        Δt = results['Δt']
        df.index = [f'{Δt * i} s' for i in range(len(results['clot size']))]
        df.columns = [f'simulation {i}' for i in range(1, N_SIMULATIONS+1)]

        df.to_excel(writer, sheet_name=f't50 = {t50}')       