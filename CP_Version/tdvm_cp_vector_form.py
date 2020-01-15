# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:44:15 2019

@author: zlibn
"""

import sys
sys.path.append("C:/Users/zlibn/Desktop/source")

import numpy as np
import tensorly as tl
import tensorly.tenalg.proximal as proximal
from  tensorly.tenalg import khatri_rao
from numpy.linalg import pinv
from sklearn.preprocessing import normalize
from numpy.linalg import norm
from numpy.linalg import inv
from math import sqrt
from Telegram_chatbot import MTRobot


class TDVMCP:
    def __init__(self, rank_init, K, alpha, beta, gamma, delta, tol):
        """
        Explanation:
        -----------
        rank_init: the initial Rank of CP-Decomposition
        K: max iterations
        alpha: station L1-norm for weak correlation
        beta: weight lambda L1-norm for low-rank
        gamma: network regularization
        tol: stopping tolerance
        """ 
        self.rank_init = rank_init
        self.max_iter = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.tol = tol
        # output needs to be added as well
        self.recons_loss_ = []
        self.lasso_on_u1_ = []
        self.lasso_on_weight_ = []
        self.tdvm_loss_ = []

    def remove_r(self, U, r):
        U_xr = []
        for i in range(len(U)):
            temp = np.delete(U[i],r,axis=1) #to delete the r-th column (r-th rank)
            U_xr.append(temp)
        return U_xr
    
    def extract_r(self, U, r):
        ulist = []
        for i in range(len(U)):
            ulist.append(U[i][:,r])
        return ulist
    
    def fit(self, X, L_POI, L_NET, Omega):
        """
        
        Input:
        ------
        X: the incomplete tensor
        L_POI: POI Similarity Laplacian Matrix
        L_NET: Network Laplacian Matrix
        Omega: Observation Indicator Tensor (Omega: A binary tensor with same shape as X, where 1s show which entries of X are missing)
        
        Output:
        -------
        X_hat: the completed X with estimated missing value
        R: the estimated rank
        U: decomposition mode matrices
        Lambda: weights
        V: U1 before softthresholding and normalization
        """
        
        ############### Initialization ###############
        # initialize U, Lambda #
        U = self.initialize_Us(X.shape)
        V = np.random.rand(X.shape[0], self.rank_init)
        Lambda = np.random.rand(self.rank_init)
        R = self.rank_init

        # Store actual values of the missing part of X in X_target so that we can track the reconstruction loss
        X_target = X[np.nonzero(Omega)].copy() # real value
        
        # Zeroize each missing variables
        X[np.nonzero(Omega)] = 0
        X_hat = X.copy() #the initial value of X_hat is X
        # First guess for missing value
        X_pred = tl.kruskal_tensor.kruskal_to_tensor(U, weights=Lambda)[np.nonzero(Omega)]
        
        ############### Interation ###############
        Y = X_hat
        # Start of Loop-0 #
        for epoch in range(self.max_iter):
            ## Start of Loop-1 ##
            for r in range(R):
                U_xr = self.remove_r(U, r)
                Lambda_xr = np.delete(Lambda,r)
                Yr = Y - tl.kruskal_tensor.kruskal_to_tensor(U_xr, weights=Lambda_xr) #residual: following Equation(5)'s definition on Xr
                ### Start condition 0 ###
                if Lambda[r] != 0:
                    #### Start of Loop-2 ####
                    for n in range(len(X.shape)):
                        Multi_Us_xUn = [x for i,x in enumerate(U) if i!=n] #exclude n-th mode
                        modes_dot_xn = np.delete(list(range(len(X.shape))),n) #exclude n-th mode dot operation
                        ##### start condition 1 #####
                        if n ==0:
                            ##### update ur^1 #####
                            list(range(len(X.shape)))
                            V[:,r] = tl.tenalg.multi_mode_dot(Yr,self.extract_r(Multi_Us_xUn, r), modes = modes_dot_xn)
                            W = inv( (Lambda[r]**2) * np.identity(X.shape[0]) + self.gamma*L_POI + self.delta*L_NET)
                            U[n][:,r] = proximal.soft_thresholding ( W.dot(Lambda[r]*V[:,r]), self.alpha/(Lambda[r]**2) )
                            
                            #print(f'after soft_thresholding u1_({r})')
                        else:
                        ##### continue condition 1 #####
                            ##### update ur^k #####
                            U[n][:,r] = tl.tenalg.multi_mode_dot(Yr,self.extract_r(Multi_Us_xUn, r), modes = modes_dot_xn)/Lambda[r]
                        ##### end condition 1 #####
                        U[n][:,r] = normalize(U[n][:,r][:,np.newaxis], axis=0).ravel() #normalization
                    #### end of Loop-2 ####
                    ### update lambda_r ###
                    Lambda[r] = proximal.soft_thresholding ( tl.tenalg.multi_mode_dot(Yr,self.extract_r(U, r)) , self.beta)
                    ### update Yr ###
                    Yr = Y - tl.kruskal_tensor.kruskal_to_tensor(U, weights=Lambda)# Yr = Yr - multi_outer_product(extraxt_r(U, r))
                else:
                    pass
                ### End condition 0 ###
            ## End of Loop-1 ##
            
            ## Update X_hat ##
            X_pred = tl.kruskal_tensor.kruskal_to_tensor(U, weights=Lambda)[np.nonzero(Omega)]
            X_hat[np.nonzero(Omega)] = X_pred.copy()
            ## Update Rank ##
            R = np.count_nonzero(Lambda)
            ## check the convergence condition ##
            #error = norm(X_hat[np.nonzero(Omega)]- X_target)/norm(X_target)
            recons_loss, tdvm_loss = self.log_losses(X_target, X_pred, U[0], L_POI, L_NET, Lambda)
            MTRobot.sendtext("Iteration: {}/{} - Rank:{} - Recons Loss:{} - TDVM Loss:{}".format(epoch+1, self.max_iter, R, round(recons_loss,2), round(tdvm_loss,2)))
            if epoch == 0:
                if recons_loss - 0 < self.tol:
                    break
            else:
                if self.recons_loss_[epoch-1] - recons_loss < self.tol:
                    MTRobot.sendtext("converged in advance")
                    break
            #print("Iteration: {}/{} - Rank:{} - Error:{}".format(epoch+1, self.max_iter, R, error))
        # End of Loop_0 #
        return X_hat, R, U, Lambda, V, recons_loss
    
    def initialize_Us(self, X_shape, init_type="CP"):
        U = []
        for n in range(len(X_shape)):
            U.append(tl.tensor(np.random.rand(X_shape[n], self.rank_init)))
        return U
    
    def log_losses(self, X_real, X_pred, U1, L_POI, L_NET, Lambda):
        recons_loss = 0.5*np.sum((X_real - X_pred) ** 2)
        lasso_on_u1 = self.alpha * norm(U1.flatten(), ord=1)
        lasso_on_weight = self.beta * norm(Lambda, ord=1)
        penalty_poi = 0.5*self.gamma*np.trace(U1.T.dot(L_POI).dot(U1))
        penalty_net = 0.5*self.delta*np.trace(U1.T.dot(L_NET).dot(U1))
        tdvm_loss = recons_loss + lasso_on_u1 + lasso_on_weight + penalty_poi + penalty_net
        self.recons_loss_.append(recons_loss)
        self.lasso_on_u1_.append(lasso_on_u1)
        self.lasso_on_weight_.append(lasso_on_weight)
        self.tdvm_loss_.append(tdvm_loss)
        return recons_loss, tdvm_loss
