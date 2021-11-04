# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 09:27:44 2020

@author: Alex
"""

import time
import numpy as np
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import data_utilities as du
import matplotlib.pyplot as plt
import graphing_utilities as gu
import tabulate
import pickle
import torch

from copy import deepcopy
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from matplotlib import cm
from torch.utils.data import DataLoader, Dataset, TensorDataset
from mpl_toolkits.mplot3d import Axes3D
from ANN_lib import MSELoss, minDevMSELoss, transferFunc, minDevMSELossCS2TM1, transferFuncCS2TM1, minDevMSELossCS2TM3, transferFuncCS2TM3
from pylab import xlabel, ylabel

rnd_seed = np.random.randint(0, 999999)
print('Random NumPy Seed :', rnd_seed)
rnd = np.random.RandomState(rnd_seed)

class Model(object):
    def __init__(self, fix_hyp, var_hyp):
        self.dtype = torch.float
        self.fix_hyp = fix_hyp
        self.var_hyp = var_hyp
        
        # Source and Transfer Model Templates and File Paths:
        self.ANN_template = self.fix_hyp['ANN_template']
        root_path   = self.fix_hyp['root_path']
        source_name = self.fix_hyp['source_name']
        model_name  = self.fix_hyp['model_name']
       
        if source_name != None: self.fix_hyp['source_path'] = root_path + '\{}'.format(source_name)
        self.fix_hyp['model_path'] = root_path + '\{}'.format(model_name)
        self.model_trials_path = root_path + '\{}_training_trials'.format(model_name)
        
        # Define Hyperparameter Search Space:
        g1 = lambda k, r: hp.uniform(k, r[0], r[1])
        g2 = lambda k, r: hp.choice(k, r)

        self.var_hyp = {k: g1(k, r) if type(r[0]) == float else g2(k, r) for k, r in var_hyp.items()}
        
    def copy(self):
        return deepcopy(self)



## ========================================================================= ##
# ----------------------- Hyperparameter Optimization ----------------------- #
## ========================================================================= ##

    def kfoldfit(self, data_trainval, new_evals=25, warm_start=False):
        hps = {**{'hyp': {**self.var_hyp, **self.fix_hyp}}, **{'trainval': data_trainval}}
        if warm_start:
            trials = torch.load(self.model_trials_path)
            max_evals = len(trials.trials) + new_evals
        else:
            trials = Trials()
            max_evals = new_evals
        fmin(self._kfold_obj_func_, hps, tpe.suggest, max_evals, trials=trials, rstate=rnd)
        torch.save(trials, self.model_trials_path)
        
        
        best_trial = trials.best_trial['result']
        self.best_hyp = best_trial['hyp']
        self.ANN = best_trial['best_ANN']
        self.RANKED_ANNs = best_trial['ranked_ANNs']
        torch.save(self.ANN, self.fix_hyp['model_path'])

        
        self.hyp_space = {k: np.array([]) for k in self.var_hyp.keys()}
        self.hyp_space['loss'] = np.array([])
        for trial in trials.trials:
            for k in self.var_hyp.keys():
                self.hyp_space[k] = np.append(self.hyp_space[k], trial['result']['hyp'][k])
            self.hyp_space['loss'] = np.append(self.hyp_space['loss'], trial['result']['loss'])
        return self
    
    
    def _kfold_obj_func_(self, hps):
        self._initialize_hyperparameters_(hps['hyp'])
        folds = hps['trainval'].shape(dn=1)
        
        history = np.zeros((self.max_epochs // self.k_size, 6, len(folds)), dtype=object)
        for n, exp in enumerate(folds):
            
            data_train, data_valid = hps['trainval'].train_test_split([exp])
            data_train, data_valid = self.train_type(data_train), self.test_type(data_valid)
            self._kfit_(data_train, data_valid)
            history[:, :, n] = self.history
        
        mean_TrLoss       = np.mean(history[:, 1, :], axis=1, keepdims=True)
        mean_VdLoss       = np.mean(history[:, 2, :], axis=1, keepdims=True)
        mean_memory       = np.mean(history[:, 4, :], axis=1, keepdims=True)
        mean_penalty      = np.mean(history[:, 5, :], axis=1, keepdims=True)
        best_epoch_index  = np.argmin(mean_VdLoss)
        ranked_fold_index = np.argsort(history[best_epoch_index, 2, :])
        best_fold_index   = ranked_fold_index[0]
        best_memory       = mean_memory[best_epoch_index]
        best_TrLoss       = mean_TrLoss[best_epoch_index]
        best_VdLoss       = mean_VdLoss[best_epoch_index]
        best_penalty      = mean_penalty[best_epoch_index]
        
        best_epoch        = history[best_epoch_index, 0, 0]
        best_mean_MAPE    = mean_VdLoss[best_epoch_index].item()
        best_ANN          = history[best_epoch_index, 3, best_fold_index]
        ranked_ANNs       = history[best_epoch_index, 3, ranked_fold_index]
        best_MAPE         = history[best_epoch_index, 2, best_fold_index]
        

        hps['hyp']['max_epochs'] = best_epoch
        return {'loss'        : best_mean_MAPE, 
                'status'      : STATUS_OK, 
                'best_ANN'    : best_ANN, 
                'best_MAPE'   : best_MAPE,
                'best_memory' : best_memory,
                'best_TrLoss' : best_TrLoss,
                'best_VdLoss' : best_VdLoss,
                'best_penalty': best_penalty,
                'ranked_ANNs' : ranked_ANNs,
                'hyp'         : hps['hyp'],
                'history'     : np.array(np.column_stack([history[:, 0, 0], mean_TrLoss, mean_VdLoss, mean_memory]), dtype=float).reshape((-1, 4, 1))}
    
    
    
## ========================================================================= ##
# --------------------------- ANN Initialization ---------------------------- #
## ========================================================================= ##
    
    def _initialize_hyperparameters_(self, hyp):
        self.hyp = hyp
        # Unpacks Hyperparameters:
        self.max_epochs = int(self.hyp['max_epochs'])
        self.lr_initial = 10 ** self.hyp['lr_initial']
        self.decay      = self.hyp['decay']
        self.hl_sizes   = self.hyp['hl_sizes'] 
        self.k_size     = self.hyp['k_size']
        self.memory     = self.hyp['memory']
        
    def _initialize_tensors_(self, data_train, data_valid):
        d0 = data_train.shape(dn=0)
        self.idx = {k: [n for n, j in enumerate(d0) if j in v] for k, v in self.xyz.items()}
        self.var = {j: k for k, j in enumerate(d0)}
        
        data_train = data_train.tensor()
        self.x_train = data_train[:, self.idx['x']]
        self.y_train = data_train[:, self.idx['y']]
        self.z_train = data_train[:, self.idx['z']]
        data_valid = data_valid.tensor()
        self.x_valid = data_valid[:, self.idx['x']]
        self.y_valid = data_valid[:, self.idx['y']]
        self.z_valid = data_valid[:, self.idx['z']]
        
    
    def _initialize_network_(self, data_train):
        torch.manual_seed(self.hyp['torch_seed'])
        self.input_size = self.hyp['input_size'] = self.x_train.size(1)
        self.output_size = self.hyp['output_size'] = self.y_train.size(1)
        self.ANN0 = deepcopy(self.ANN_template)(self.hyp) #!!!
        self.ANN0.initParams()
        self.ANN0.resetSource()
   
    def _initialize_optimizer_(self, ANN):
        self.optimizer = optim.Adam(ANN.parameters(), lr=self.lr_initial, weight_decay=self.decay)
        if self.hyp['loss_fn'] == 'MSELoss':
            self.loss_fn = MSELoss()
            self.transfer_func = lambda a, b, c: 0
        elif self.hyp['loss_fn'] == 'minDevMSELoss':
            source_model = torch.load(self.hyp['source_path'])
            self.loss_fn = minDevMSELoss(source_model.linears, self.memory)
            self.transfer_func = transferFunc(source_model.linears, self.memory)
        elif self.hyp['loss_fn'] == 'minDevMSELossCS2TM3':
            source_model = torch.load(self.hyp['source_path'])
            self.loss_fn = minDevMSELossCS2TM3(source_model.linears, self.memory)
            self.transfer_func = transferFuncCS2TM3(source_model.linears, self.memory)
        elif self.hyp['loss_fn'] == 'minDevMSELossCS2TM1':
            source_model = torch.load(self.hyp['source_path'])
            self.loss_fn = minDevMSELossCS2TM1(source_model.linears, self.memory)
            self.transfer_func = transferFuncCS2TM1(source_model.linears, self.memory)

    
    
## ========================================================================= ##
# --------------------------- ANN Training Loop ----------------------------- #
## ========================================================================= ## 
     
    def _kfit_(self, data_train, data_valid):
            
        self.data_train = data_train
        self.data_valid = data_valid #!!!

        self.exp_valid = self.data_valid.shape(dn=1) #!!!
        self._initialize_tensors_(data_train, data_valid)
        self._initialize_network_(data_train)
        
        self.ANN = deepcopy(self.ANN0)
        self.ANN0 = None #!!!
        self._initialize_optimizer_(self.ANN)

        # --- Iterates over Epochs to Fit Network --- #   
        self.history = np.zeros((self.max_epochs // self.k_size, 6), dtype=object)
        #t0 = time.clock()
        for k in range(self.max_epochs // self.k_size):
            for epoch in range(self.k_size):

                # Reset Gradients:
                self.optimizer.zero_grad()
                
                # Training Loss:
                u = self.ANN(self.x_train)
                y_train_pred = self.prior(self.x_train, u, self.z_train)
                loss = self.loss_fn(y_train_pred, self.y_train, self.ANN.linears)
                
                # Step Training:
                loss.backward()
                self.optimizer.step()
                self.ANN.resetSource()
    
            # Validation Error:
            self.history[k, 0] = k * self.k_size
            self.history[k, 1] = float(loss)
            self.history[k, 2] = float(self.loss_fn(self.ANN(self.x_valid), self.y_valid, self.ANN.linears))
            self.history[k, 3] = self.ANN
            self.history[k, 4], self.history[k, 5] = self.transfer_func(y_train_pred, self.y_train, self.ANN.linears)

            
            
            
## ========================================================================= ##
# -------------------------------- Simulation ------------------------------- #
## ========================================================================= ## 
    
    def simulate(self, X0_DataSet, steps=None):
        d0, d1, d2, _ = X0_DataSet.shape()
        idx = {k: [n for n, j in enumerate(d0) if j in v] for k, v in self.xyz.items()}
        var = {j: k for k, j in enumerate(d0)}
        X0_DataSet = X0_DataSet[:, :, :, 0]
        wid, xid, yid, zid = idx['w'], idx['x'], idx['y'], idx['z']
        X = {}
        ex_X = {}
        with torch.no_grad():
            for i in d1:
                if steps is None: steps = range(d2[i] - 1)
                X0 = X0_DataSet[:, i].tensor()
                ex_X0 = deepcopy(X0)
                for n in steps:
                    x = X0[n:n + 1, xid]
                    z = X0[n:n + 1, zid]

                    u = self.ANN(x)
                    y = self.prior(x, u, z)
                    X0[n + 1, wid] = self.step(x, y, z)
                    ex_X0[n + 1, wid] = self.step(ex_X0[n:n+1, xid], ex_X0[n:n+1, yid], ex_X0[n:n+1, zid])
                
                X[i] = {j: np.array(X0[:, i]).reshape((-1, 1, 1)) for j, i in var.items()}
                ex_X[i] = {j: np.array(ex_X0[:, i]).reshape((-1, 1, 1)) for j, i in var.items()}
        X_DataSet = du.DataSet(X)
        ex_X = du.DataSet(ex_X)
        return X_DataSet, ex_X
    
    def bagsimulate(self, X0_DataSet, pop_size=10):
        X0_DataSet = self.test_type(X0_DataSet)
        d0, d1, d2, _ = X0_DataSet.shape()
        idx = {k: [n for n, j in enumerate(d0) if j in v] for k, v in self.xyz.items()}
        jdx = {j: k for k, j in enumerate(d0)}
        X0_DataSet = X0_DataSet[:, :, :, 0]
        wid, xid, yid, zid = idx['w'], idx['x'], idx['y'], idx['z']
        ub_X = {}; me_X = {}; lb_X = {}; ex_X = {}
        with torch.no_grad():
            for i in d1:
                
                x = deepcopy(X0_DataSet[:, i].tensor().reshape(d2[i], -1, 1))
                X0 = torch.cat([deepcopy(x)] * pop_size, dim=2)
                ub_X0 = deepcopy(x)
                me_X0 = deepcopy(x)
                lb_X0 = deepcopy(x)
                ex_X0 = deepcopy(x)
                
                for t in range(d2[i] - 1):
                    aggregated_X1 = []
                    for n in range(pop_size):
                        X1 = X0[t + 1, :, n]
                        for ANN in self.RANKED_ANNs:
                            x = X0[t:t + 1, xid, n] #!!!
                            z = X0[t:t + 1, zid, n] #!!!
                            u = ANN(x)
                            y = self.prior(x, u, z)
                            X1[wid] = self.step(x, y, z)
                            aggregated_X1.append(deepcopy(X1.reshape(1, -1, 1)))
                            
                    aggregated_X1 = torch.cat(aggregated_X1, dim=2)
                    cov  = np.cov(aggregated_X1[0, idx['w'], :].numpy())
                    var  = torch.tensor(np.diagonal(cov, 0), dtype=self.dtype).reshape((1, -1, 1))
                    mean = np.mean(aggregated_X1[0, idx['w'], :].numpy(), axis=1)
                    norm = rnd.multivariate_normal(mean, cov, size=pop_size - 1)
                    
                    mean = torch.tensor(mean.reshape((1, -1)), dtype=self.dtype)
                    norm = torch.tensor(norm, dtype=self.dtype)
                    X1 = torch.cat([mean, norm], dim=0)
                    X1 = X1.transpose(0, 1).reshape((1, -1, pop_size))
                    X0[t + 1, wid, :] = X1
                    
                    
                    ub_X0[t + 1, wid] = deepcopy(X1[:, :, 0].reshape((-1, 1)) + 2 * var ** 0.5)
                    me_X0[t + 1, wid] = deepcopy(X1[:, :, 0].reshape((-1, 1)))
                    lb_X0[t + 1, wid] = deepcopy(X1[:, :, 0].reshape((-1, 1)) - 2 * var ** 0.5)
                    ex_X0[t + 1, wid] = self.step(ex_X0[t:t+1, xid], ex_X0[t:t+1, yid], ex_X0[t:t+1, zid])
                    
                
                ub_X[i] = {j: np.array(ub_X0[:, i, :]).reshape((-1, 1, 1)) for j, i in jdx.items()}
                me_X[i] = {j: np.array(me_X0[:, i, :]).reshape((-1, 1, 1)) for j, i in jdx.items()}
                lb_X[i] = {j: np.array(lb_X0[:, i, :]).reshape((-1, 1, 1)) for j, i in jdx.items()}
                ex_X[i] = {j: np.array(ex_X0[:, i, :]).reshape((-1, 1, 1)) for j, i in jdx.items()}
                
        ub_X = du.DataSet(ub_X)
        me_X = du.DataSet(me_X)
        lb_X = du.DataSet(lb_X)
        ex_X = du.DataSet(ex_X)
        return ub_X, me_X, lb_X, ex_X

    
## ========================================================================= ##
# ----------------------------- Error Evaluation ---------------------------- #
## ========================================================================= ## 
   
    def _evaluate_offline_error_(self, data_valid, error_func):
        data_valid = data_valid[:, :, :, 0]
        sim_data, data_valid = self.simulate(data_valid)
        data_valid = du.invscale(data_valid, self.stats)[self.xyz['w']]
        sim_data = du.invscale(sim_data, self.stats)[self.xyz['w']]
        error = error_func(data_valid, sim_data)
        d0, d1, _, _ = error.shape()
        mean_error = np.mean([error.X[i][j] for j in d0 for i in d1])
        return mean_error
    
    def _evaluate_offline_MAPE_(self, data_valid):
        error_func = lambda meas, simu: 100 * du.mean(du.absolute((meas - simu) / meas)[:, :, 1:], axis=0)
        mean_MAPE = self._evaluate_offline_error_(data_valid, error_func)
        return mean_MAPE
    
    def _evaluate_offline_MAE_(self, data_valid):
        error_func = lambda meas, simu: 100 * du.amax(du.absolute((meas - simu) / meas)[:, :, 1:], axis=0)
        mean_MAPE = self._evaluate_offline_error_(data_valid, error_func)
        return mean_MAPE
    
    def _evaluate_offline_RPD_(self, data_valid):
        error_func = lambda meas, simu: 100 * du.amax(du.absolute(2 * (meas - simu) / (meas + simu))[:, :, 1:], axis=0)
        mean_MAPE = self._evaluate_offline_error_(data_valid, error_func)
        return mean_MAPE
    
    def accuracy_table(self, ub_X, me_X, lb_X, ex_X, output=False):
        d0, d1, d2, d3 = me_X.shape()
        
        lb = lambda i, j: lb_X.X[i][j][:, :, 0]
        ub = lambda i, j: ub_X.X[i][j][:, :, 0]
        me = lambda i, j: me_X.X[i][j][:, :, 0]
        ex = lambda i, j: ex_X.X[i][j][:, :, 0]
        

        MRPE = {i: {j: 200 * np.mean(np.absolute((ex(i, j) - me(i, j)) / (ex(i, j) + me(i, j))), axis=0) for j in d0} for i in d1}
        MRPE_X = np.mean([MRPE[i]['CX'] for i in d1])
        MRPE_N = np.mean([MRPE[i]['CN'] for i in d1])
        MRPE_P = np.mean([MRPE[i]['CP'] for i in d1])
        
        MAE = {i: {j: np.mean(np.absolute(ex(i, j) - me(i, j)), axis=0) for j in d0} for i in d1}
        MAE_X = np.mean([MAE[i]['CX'] for i in d1])
        MAE_N = np.mean([MAE[i]['CN'] for i in d1])
        MAE_P = np.mean([MAE[i]['CP'] for i in d1])
        
        MRSD = {i: {j: 100 * np.mean(np.absolute((ub(i, j) - lb(i, j)) / (ub(i, j) + lb(i, j))), axis=0) / 2 for j in d0} for i in d1}
        MRSD_X = np.mean([MRSD[i]['CX'] for i in d1])
        MRSD_N = np.mean([MRSD[i]['CN'] for i in d1])
        MRSD_P = np.mean([MRSD[i]['CP'] for i in d1])
        
        MSD = {i: {j: np.mean(np.absolute(ub(i, j) - lb(i, j)), axis=0) / 2 for j in d0} for i in d1}
        MSD_X = np.mean([MSD[i]['CX'] for i in d1])
        MSD_N = np.mean([MSD[i]['CN'] for i in d1])
        MSD_P = np.mean([MSD[i]['CP'] for i in d1])
        
        
        if output: 
            return MAE_X, MAE_N, MAE_P, MSD_X, MSD_N, MSD_P
        else:
            print('==========================================================================================')
            print('==========================================================================================')
            print('(X)  MRPE: {:.1f},     MAE: {:.3f},     MRSD: {:.1f},     MSD: {:.3f}'.format(MRPE_X,  MAE_X, MRSD_X, MSD_X))
            print('(N)  MRPE: {:.1f},     MAE: {:.0f},     MRSD: {:.1f},     MSD: {:.0f}'.format(MRPE_N,  MAE_N, MRSD_N, MSD_N))
            print('(P)  MRPE: {:.1f},     MAE: {:.3f},     MRSD: {:.1f},     MSD: {:.1f}'.format(MRPE_P,  MAE_P, MRSD_P, MSD_P))
            print('==========================================================================================')
            print('==========================================================================================')
        

    
    
## ========================================================================= ##
# --------------------------- Debugging Functions --------------------------- #
## ========================================================================= ## 

    def display_ANN_params(self):
        for layer in self.ANN.linears:
            weights = layer.weight.data
            biases = np.array(layer.bias.data).reshape((-1, 1))
            sns.heatmap(weights, center=0, cmap='vlag')
            plt.show()
            sns.heatmap(biases, center=0, cmap='vlag')
            plt.show()
            
    def display_ANN0_params(self):
        for layer in self.ANN0.linears:
            weights = layer.weight.data
            biases = np.array(layer.bias.data).reshape((-1, 1))
            sns.heatmap(weights, center=0, cmap='vlag')
            plt.show()
            sns.heatmap(biases, center=0, cmap='vlag')
            plt.show()
    
    def plt_hyp_surface(self, xp, yp):
        x = self.hyp_space[xp]
        y = self.hyp_space[yp]
        z = self.hyp_space['loss']
        fig, ax = plt.subplots(nrows=1)
        ax.tricontour(x, y, z, levels=30, linewidths=0, colors='k')
        cntr2 = ax.tricontourf(x, y, z, levels=30, cmap="RdBu_r")
        fig.colorbar(cntr2, ax=ax)
        ax.plot(x, y, 'ko', ms=3)
        ax.set_title('Loss Function')
        plt.subplots_adjust(hspace=0.5)
        xlabel(xp)
        ylabel(yp)
        plt.show()
        
    def plt_hyp_space(self, xp):
        x = self.hyp_space[xp]
        xi = np.argsort(x)
        y = self.hyp_space['loss']
        plt.plot(x[xi], y[xi])
        xlabel(xp)
        plt.show()
        
        
    def display_learning_curve(self):
        X1 = {'Training Curves': {'EP': self.history[:, 0:1], 'LS': self.history[:, 1:2, :]}}
        X2 = {'Training Curves': {'EP': self.history[:, 0:1], 'LS': self.history[:, 2:3, :]}}
        X3 = {'Training Curves': {'EP': self.history[:, 0:1], 'LS': self.history[:, 3:, :]}}
        X1, X2, X3, = du.DataSet(X1), du.DataSet(X2), du.DataSet(X3)

        gu.plot_dataset(plts=[{'data': X1, 'line': 'b-', 'label': 'Training Loss'},
                              {'data': X2, 'line': 'r-', 'label': 'Validation Loss'},
                              {'data': X3, 'line': 'g-', 'label': 'Transferability'}],
                        x_axis='EP', y_axis='LS', x_label='Training Epoch', y_label='MSE Loss')

        
        
        
        
        
            
            
            
            
            
            