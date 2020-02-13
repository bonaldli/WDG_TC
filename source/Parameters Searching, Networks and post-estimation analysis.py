# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 13:41:12 2019

@author: zlibn
"""
"""
Important: Most of modules (no matter already-existing or self-written modules), 
here the setting environment is incomplete, so please run file 'CP_version.py' first,
then run this file
"""

# In[]
""" POI Similarity Netowrk"""
import pandas as pd
import numpy as np
POI = pd.read_csv("C:\\Users\\zlibn\\Desktop\\POI.CSV") #only the selected key
poi = POI.set_index('POI')
poi_sim = sim_matrix.cal(poi)

# In[] 
"""To the the connection network:
   Connected(K-Hop Reachable, K=5):    aij = 1
   Not-Connected(Not K-Hop Reachable): aij = 0
"""
net = np.zeros((15,15))
i = [[0],[1],[1] ,[2],[4] ,[5],[5], [5], [5], [6], [8],[10],[10],[11],[12]]
j = [[7],[7],[12],[3],[14],[8],[10],[11],[12],[8],[12],[11],[12],[12],[13]]
net[i,j] = 1
inds = np.triu_indices_from(net,k=1)
net[(inds[1], inds[0])] = net[inds]

# In[]
""" search for best tuning parameters: (alpha, beta) """
error_list = np.zeros((11,11))
i = 0
#search for the best parameter combination
for alpha in range(0, 110, 10):
    temp = []
    for beta in range(1000, 2100, 100):
        #print(f'alpha: {alpha} / beta: {beta}')
        X_hat = inflow.copy()
        solver = TDVMCP(rank_init, K=max_iter, alpha = alpha, beta = beta, gamma=0, tol = tol)
        X_hat, R, U, Lambda, V, error = solver.fit(X_hat, L_POI, L_NET, Ohm)
        temp.append(error)
    error_list[i,:] = temp
    print(f'finish alpha{alpha}')
    i = i + 1
# analysis the 2D search result of best parameter#
error_list = pd.DataFrame(error_list)
error_list.columns = ["alpha:{0}".format(i) for i in range(0, 110, 10)]
error_list.index = ["beta:{0}".format(i) for i in range(1000, 2100, 100)]

#get the best alpha on a specific beta
best_alphabeta = pd.concat([error_list.idxmin(axis = 1) , np.amin(error_list, axis  = 1)], axis =1)

# In[]
""" search for the best gamma for a given (alpha, beta)"""
temp = []
for delta in range(0,110, 10):
    X_hat = inflow.copy()
    solver = TDVMCP(rank_init, K=60, alpha = 30, beta = 1300, gamma = 90, delta = delta, tol = 1000)
    X_hat, R, U, Lambda, V, error = solver.fit(X_hat, L_POI, L_NET, Ohm)
    temp.append(error)
    print(f'finish delta{delta}')

# In[]
"""search for the best delta for given (alpha, beta, gamma)s"""
albe = np.zeros((11,3))
albe[:,0] = [50,60,70,30,80,80,30,60,80,30,70] # alpha
albe[:,1] = list(range(1000, 2100, 100)) # beta
albe[:,2] = [70,100,20,0,50,70,10,70,80,100,30]
error_list_3 = np.zeros((11,11))
for i in range(albe.shape[0]):
    temp = []
    for delta in range(0,110,10):
        X_hat = inflow.copy()
        solver = TDVMCP(rank_init=150, K=60, alpha = albe[i,0], beta = albe[i,1], gamma = albe[i,1], delta = delta, tol = 1000)
        X_hat, R, U, Lambda, V, error = solver.fit(X_hat, L_POI, L_NET, Ohm)
        temp.append(error)
    error_list_3[:,i] = temp #column is alpha-beta; row is gamma
    print(f'finish albe:{i}')
    i = i + 1

# analysis the 2D search result of best parameter#
error_list_3 = pd.DataFrame(error_list_3)
error_list_3.columns = ["alpha_beta_gamma:{0}".format(i) for i in range(11)]
error_list_3.index = ["delta:{0}".format(i) for i in range(0,110,10)]

#get the best alpha on a specific beta
best_gamma = pd.concat([error_list_3.idxmin(axis = 0) , np.amin(error_list_3, axis  = 0)], axis =1)
#                       axi = 0 min of each column       axis=0: min of each column

# In[]:
<<<<<<< HEAD:source/Parameters Searching, Networks and post-estimation analysis.py
""" post-estimation descriptive analysis"""
# [1] Clustering
rank = np.nonzero(Lambda)[0]; # to get the non-zero rank
U0_nz = U0[:,rank];
# then jump to "pca_plus_hierarchical_6PC.py"

# [2] Check Correlation

def correlation_matrix_1(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm
    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(111)
    ax1.xaxis.tick_top()
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap) #Compute pairwise correlation of columns
    ax1.grid(True)
    plt.title('Correlation Matrix')
    plt.xticks(np.arange(15), A_m_stn)
    plt.yticks(np.arange(15), A_m_stn)
    for tick in ax1.xaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(12)
    for tick in ax1.yaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(12)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[-0.1,0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
    plt.show()

U0_nz_df = pd.DataFrame(np.transpose(U0_nz))
correlation_matrix_1(U0_nz_df)
# count how many is strongly-correlated
corr = U0_nz_df.corr()
N_stn = len(A_m_stn)
weakly_correlated_rate = 1 - (sum(corr[corr > 0.7].count()) - N_stn)/(N_stn**2 - N_stn)

# In[]:
""" Uniform Comparison for Recon_MSE and MAPE"""
# Input the last day's full prediction (15, 247)
def mape_vectorized_v2(a, b): 
    mask = a != 0
    return (np.fabs(a - b)/a)[mask].mean()

X_real = A_m[:, 74:247, 50]
X_pred = tdvm_cp[:, 74:247]
0.5*np.sum((X_real - X_pred) ** 2)

mape_vectorized_v2(np.squeeze(X_real), np.squeeze(X_pred))









=======
>>>>>>> 61a92ff74a2b9cc98a424bb9715b5842854556d0:CP_Version/Parameters Searching and Networks.py
