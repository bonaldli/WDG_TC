# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:22:21 2019

@author: zlibn
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# In[1] 
"""Input U0 with 0-rank columns deleted"""
inputdata = U0_nz # the station mode 
X = pd.DataFrame(inputdata)
X.columns = ["Rank{0}".format(i) for i in range(0,inputdata.shape[1])]
from sklearn.preprocessing import StandardScaler
#X_std = StandardScaler().fit_transform(X)
#X_std = pd.DataFrame(X_std)
#X_std.columns = ["Rank{0}".format(i) for i in range(1,51)]
s = A_m_stn # the inflow station code, inflow_stn_namecsv
X = X.set_index(s)
# In[2] 
"""PCA to decrease the dimension"""
from sklearn.decomposition import PCA
pca = PCA(n_components=6)
pca.fit(X)
PCA(copy=True, n_components=6, whiten=False)
#This gives us an object we can use to transform our data by calling transform.
X_6d = pca.transform(X)

X_6d = pd.DataFrame(X_6d)
X_6d.index = X.index
X_6d.columns = ['PC1','PC2','PC3','PC4','PC5','PC6']
X_6d.head()

print(pca.explained_variance_ratio_)
#We see that the first PC already explains almost 32% of the variance
# In[3] 
"""To plot the explanined variance"""
#Calculating Eigenvecors and eigenvalues of Covariance matrix
mean_vec = np.mean(X, axis=0)
cov_mat = np.cov(X.T) #or X_std
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# Create a list of (eigenvalue, eigenvector) tuples
eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort from high to low
eig_pairs.sort(key = lambda x: x[0], reverse= True)

# Calculation of Explained Variance from the eigenvalues
tot = sum(eig_vals)
var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance
cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance

# PLOT OUT THE EXPLAINED VARIANCES SUPERIMPOSED 
plt.figure(figsize=(10, 5))
plt.bar(range(len(var_exp)), var_exp, alpha=0.3333, align='center', label='individual explained variance', color = 'g')
plt.step(range(len(cum_var_exp)), cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')

describe = X.describe()
plt.show()
# In[4] 
"""Hierarchical Clustering"""
import cmath as math
import sys
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

X_6d_PC = X_6d.iloc[:,0:6]

###### use weighted as linkage##########
mtd = 'weighted'
Z = sch.linkage(X_6d_PC, method = mtd)
fig, axes = plt.subplots(1, 1, figsize=(10, 5))
den = sch.dendrogram(Z, labels = X_6d_PC.index, leaf_font_size=10)
plt.title('Dendrogram for the clustering of stations based on ' + mtd)
plt.xlabel('Station Code')
plt.ylabel('Euclidean distance with dimensions PC1-PC6')
file_path = "C:\\Users\\zlibn\\Desktop\\dendrogram"
plt.savefig(file_path, dpi = (300))   
plt.show()
