# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 13:05:22 2025

@author: vf926215
"""

import pickle

# choose zoom which must be an integer, fyi original grid spacing is 1um
scale = 1
# choose CFL which will determine the ts
CFL = 0.8

res = pickle.load(open(f'trajectories for scale = {scale} and CFL = {CFL}.pkl', 'rb'))