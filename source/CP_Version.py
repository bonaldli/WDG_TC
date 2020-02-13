# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:26:02 2019
@author: zlibn
"""

import sys
sys.path.append("C:/Users/zlibn/Desktop/source")
import numpy as np
import pandas as pd

# needed to run again everytime "tdvm_cp_vector_form" is modified
import tdvm_cp_vector_form
from tdvm_cp_vector_form import TDVMCP
#import tdvm_cp
#from tdvm_cp import TDVMCP
from POI import sim_matrix
from scipy.sparse import csgraph

file_path = "C:/Users/zlibn/Desktop/data//processed/"
inflow = np.load(file_path+"A_m.npy")
Ohm = np.load(file_path+"ohm_m.npy")
POI = np.load(file_path+"poi_sim.npy")
NET = np.load(file_path+"net.npy")


# calculate Laplacian Matrix
L_POI = csgraph.laplacian(POI, normed=False)
L_NET = csgraph.laplacian(NET, normed=False)
#inflow = np.transpose(inflow, (2, 1, 0))
# inflow /= inflow.max()

# implementation setting #
# For the best parameter searching, check file: 'Parameters Searching and Networks.py'
rank_init = 150 #round(0.5*mean(I1, I2, I3))
max_iter = 60
alpha = 30
beta = 1600
gamma = 50
delta = 0
tol = 1000

# data input and soltion #
X_hat = inflow.copy()
solver = TDVMCP(rank_init, K=max_iter, alpha = alpha, beta = beta, gamma = gamma, tol = tol)
X_hat, R, U, Lambda, V, recons_loss = solver.fit(X_hat, L_POI, L_NET, Ohm)

# prediction result #
import matplotlib.pyplot as plt
stn = 0
plt.plot(inflow[stn,:,50])
plt.plot(X_hat[stn,:,50])

# MSE Checking
from sklearn.metrics import mean_squared_error
mse = []
for stn in range(inflow.shape[0]):
    mse.append(mean_squared_error(inflow[stn,:,50], X_hat[stn,:,50]))
