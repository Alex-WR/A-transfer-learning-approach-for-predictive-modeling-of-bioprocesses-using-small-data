# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 18:07:19 2020

@author: Alex Rogers
"""

import pandas as pd
import numpy as np
import torch
import operator 

from torch.utils.data import Dataset
from copy import deepcopy
from xlsxwriter.utility import xl_cell_to_rowcol

rnd_seed = 0
torch.manual_seed(rnd_seed)
rnd = np.random.RandomState(rnd_seed)

class DataSet(object):
    def __init__(self, X={}):
        self.dtype = torch.float
        self.X = X
        
    def copy(self):
        return deepcopy(self)
        
    def __str__(self):
        return str(self.X)
    
    def _interpret_index_(self, index):
        """
        Parameters
        ----------
        index : Slice
            First element = Column Keys, accepts strings, lists of strings or ":".
            Second element = Data Group Keys, accepts lists of hashables or ":".
            Third element = Row Index, accepts integers, indexed slices or ":".
            Fourth element = Layer Index, accepts integers, indexed slices or ":".
            
            If an undefined element follows a defined element the undefined
            element will default to ":", allowing for example, a slice to be 
            taken only over the first element without the need to specify the rest.

        Returns
        -------
        i0 : List
            List of Column Keys.
        i1 : List
            List of Data Group Keys.
        i2 : List
            List of Row Indexes.
        i3 : List
            List of Layer Indexes.
        """
        none = slice(None, None, None)
        if type(index) is slice:
            i0, i1, i2, i3 = none, none, none, none
        else:
            try:
                test = type(''.join(index))
            except:
                test = 0
            if test is str:
                i0, i1, i2, i3 = index, none, none, none
            elif len(index) == 1:
                i0, i1, i2, i3 = index[0], none, none, none
            elif len(index) == 2:
                i0, i1, i2, i3 = index[0], index[1], none, none
            elif len(index) == 3:
                i0, i1, i2, i3 = index[0], index[1], index[2], none
            elif len(index) == 4:
                i0, i1, i2, i3 = index[0], index[1], index[2], index[3]
        if type(i1) is slice:
            i1 = list(self.X.keys())[i1]
        elif type(i1) is not tuple and type(i1) is not list:
            i1 = [i1]
        
        if type(i0) is slice:
            i0 = list(self.X[i1[0]].keys())
        if type(i0) is not tuple and type(i0) is not list:
            i0 = [i0]
        
        if type(i2) is int:
            if i2 < 0: 
                i2 = slice(i2, None, None)
            else:
                i2 = slice(i2, i2 + 1)
            
        if type(i3) is int:
            i3 = slice(i3, i3 + 1) 

        return i0, i1, i2, i3
    
    def __setitem__(self, index, value):
        A, B = self, value
        d0, d1, d2, d3 = A._interpret_index_(index)
        
        for i, Bi in zip(d1, B.X.keys()):
            if i not in list(A.X.keys()):
                A.X[i] = {}
            for j, Bj in zip(d0, B.X[Bi].keys()):
                if j not in list(A.X[i].keys()):
                    A.X[i][j] = B.X[Bi][Bj]
                else:
                    A.X[i][j][d2, :, d3] = B.X[Bi][Bj]
                    
    def __getitem__(self, index):
        i0, i1, i2, i3 = self._interpret_index_(index)
        g = lambda i: {j: self.X[i][j][i2, :, i3] for j in i0}
        X = {i: g(i) for i in i1}
        return DataSet(X)
    
    def __or__(self, other):
        A, B = self, other
        Ad0, Ad1, _, _ = A.shape()
        Bd0, Bd1, _, _ = B.shape()
        d0 = Ad0 + [j for j in Bd0 if j not in Ad0]
        d1 = Ad1 + [j for j in Bd1 if j not in Ad1]
        g = lambda i: {j: B.X[i][j] if j in Bd0 else A.X[i][j] for j in d0}
        X = {i: g(i) if i in Bd1 else A.X[i] for i in d1}
        return DataSet(X)
    
    def bin_op(self, other, f):
        if type(other) is DataSet:
                
            A, B = self, other
            Ad0, Ad1, _, _ = A.shape()
            Bd0, Bd1, _, _ = B.shape()
                
            # Matching of All Exp but Forced Pairing of Potentialy Different Var:
            if Ad1 == Bd1 and (len(Ad0) == 1 and len(Bd0) == 1):
                g = lambda i: {Aj: f(A.X[i][Aj], B.X[i][Bj]) for Aj, Bj in zip(Ad0, Bd0)}
                X = {i: g(i) for i in Ad1}
            
            # Matching of All Exp and Filtered Matching of Select Var:
            elif Ad1 == Bd1 and (len(Ad0) != 1 or len(Ad0) != 1):
                if len(Bd1) > len(Ad1):
                    A, Ad0, Ad1, B, Bd0, Bd1 = B, Bd0, Bd1, A, Ad0, Ad1
                g = lambda i: {j: f(A.X[i][j], B.X[i][j]) if j in Bd0 else A.X[i][j] for j in Ad0}
                X = {i: g(i) for i in Ad1}
                    
            # Mapping Over All Exp and Filtered Matching of Select Var:
            elif len(Ad1) == 1 or len(Bd1) == 1:
                if len(Bd1) > len(Ad1):
                    A, Ad0, Ad1, B, Bd0, Bd1 = B, Bd0, Bd1, A, Ad0, Ad1
                g = lambda i: {j: f(A.X[i][j], B.X[list(Bd1)[0]][j]) if j in Bd0 else A.X[i][j] for j in Ad0}
                X = {i: g(i) for i in Ad1}
                    
        else:
            X = {i: {j: f(var, other) for j, var in exp.items()} for i, exp in self.X.items()}
        
        return DataSet(X)
    
    def relabel(self, new_var):
        d0, d1, _, _, = self.shape()
        X = {i: {var: self.X[i][j] for j, var in zip(d0, new_var)} for i in d1}
        return DataSet(X)
    
    def unary_op(self, f):
        X = {i: {j: f(var) for j, var in exp.items()} for i, exp in self.X.items()}
        return DataSet(X)
    
    def __add__ (self, other):
        return self.bin_op(other, operator.add)
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__ (self, other):
        return self.bin_op(other, operator.sub)
    def __rsub__(self, other):
        return self.__sub__(other)
    
    def __mul__ (self, other):
        return self.bin_op(other, operator.mul)
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__ (self, other):
        return self.bin_op(other, operator.truediv)
    def __rtruediv__(self, other):
        return self.__truediv__(other)
    
    def __pow__ (self, other):
        return self.bin_op(other, operator.pow)
    def __rpow__(self, other):
        return self.__pow__(other)
    
    def __and__ (self, other):
        return DataSet({**other.X, **self.X})
    
    def __neg__ (self):
        return self.unary_op(operator.neg)
    def __rneg__(self, other):
        return self.__neg__(other)
    
    def shape(self, dn=None):
        d1 = list(self.X.keys())
        if d1 == list():
            d0, d1, d2, d3 = None, None, None, None
        else:
            d0 = list(self.X[list(d1)[0]].keys())
            d2 = {i: self.X[i][d0[0]].shape[0] for i in d1}
            d3 = {i: self.X[i][d0[0]].shape[2] for i in d1}
            
        if dn is None:
            return d0, d1, d2, d3
        elif dn == 0:
            return d0
        elif dn == 1:
            return d1
        elif dn == 2:
            return d2
        elif dn == 3:
            return d3
    
    def exp_merge(self):
        d0, d1, _, _ = self.shape()
        Y = {j: np.row_stack([self.X[i][j] for i in d1]) for j in d0}
        return DataSet({0: Y})
    
    def var_merge(self):
        d0, d1, _, _ = self.shape()
        X = {i: np.column_stack([self.X[i][j] for j in d0]) for i in d1}
        return DataSet(X)
    
    def rep_stack(self, other):
        d0, d1, _, _ = self.shape()
        X = {i: {j: np.dstack((self.X[i][j], other.X[i][j])) for j in d0} for i in d1}
        return DataSet(X)
    
    def train_test_split(self, test_exps):
        d1 = self.shape(dn=1)
        trainval_exps = [i for i in d1 if i not in test_exps]
        return self[:, trainval_exps], self[:, test_exps]

    def tensor(self):
        data = self.var_merge()
        g = lambda v: np.row_stack(np.transpose(v, axes=(2, 0, 1))) 
        data = [g(v) if v.ndim == 3 else v for v in data.X.values()]
        data = np.row_stack(data)
        data = torch.from_numpy(data).type(self.dtype)
        return data
    
    def zerovars(self, new_vars):
        d0, d1, d2, d3 = self.shape()
        self = self | zeros(shape=[new_vars, d1, d2, d3])
        return DataSet(self.X)


class PyTorch_DataSet(Dataset):
    def __init__(self, x_tensor, y_tensor, z_tensor):
        self.x = x_tensor
        self.y = y_tensor
        self.z = z_tensor
        
    def __getitem__(self, index):
        return (self.x[index], self.y[index], self.z[index])

    def __len__(self):
        return len(self.x)

def rep_stack(other):
    d0, d1, _, _ = other[0].shape()
    g = lambda i, j: [other[k].X[i][j] for k in range(len(other))]
    X = {i: {j: np.dstack(g(i, j)) for j in d0} for i in d1}
    return DataSet(X)

def row_stack(other):
    d0, d1, _, _ = other[0].shape()
    g = lambda i, j: [other[k].X[i][j] for k in range(len(other))]
    X = {i: {j: np.row_stack(g(i, j)) for j in d0} for i in d1}
    return DataSet(X)

def exp(other):
    return DataSet.unary_op(other, np.exp)

def zeros(shape):
    d0, d1, d2, d3 = shape
    if type(d2) == int: d2 = {i: d2 for i in d1}
    if type(d3) == int: d3 = {i: d3 for i in d1}
    data = {i: {j: np.zeros(shape=(d2[i], 1, d3[i])) for j in d0} for i in d1}
    return DataSet(data)

def ones(shape):
    d0, d1, d2, d3 = shape
    if type(d2) == int: d2 = {i: d2 for i in d1}
    if type(d3) == int: d3 = {i: d3 for i in d1}
    data = {i: {j: np.ones(shape=(d2[i], 1, d3[i])) for j in d0} for i in d1}
    return DataSet(data)

def normal(shape, loc=0, scale=1, rnd_seed=0):
    d0, d1, d2, d3 = shape
    if type(d2) == int: d2 = {i: d2 for i in d1}
    if type(d3) == int: d3 = {i: d3 for i in d1}
    data = {i: {j: rnd.normal(loc, scale, size=(d2[i], 1, d3[i])) for j in d0} for i in d1}
    return DataSet(data)

def read_excel(excel_path, data_cells):
    xls = pd.ExcelFile(excel_path)
    X = {}
    i = 1
    for sheet, exps_cells in data_cells.items():
        df = pd.read_excel(xls, sheet)
        for exp_cells in exps_cells:
            ay, ax = xl_cell_to_rowcol(exp_cells[0])
            by, bx = xl_cell_to_rowcol(exp_cells[1])           
            x_range = range(ax, bx + 1)
            y_range = range(ay - 1, by)
            exp_data = df.iloc[y_range, x_range].to_numpy()
            f = lambda x: x.astype(float).reshape((-1, 1, 1))
            exp_data = {exp_data[0, i]: f(exp_data[1:, i]) for i in range(len(x_range))} 
            X[i] = exp_data
            i += 1
    return DataSet(X)

def _dim_npfunc_(f, other, axis):
    d0, d1, _, _ = other.shape()
    X = {i: {j: f(other.X[i][j], axis) for j in d0} for i in d1}
    return DataSet(X)  

def _npfunc_(f, other):
    d0, d1, _, _ = other.shape()
    X = {i: {j: f(other.X[i][j]) for j in d0} for i in d1}
    return DataSet(X)

def amin(other, axis):
    f = lambda a, axis: np.amin(a, axis, keepdims=True)
    return _dim_npfunc_(f, other, axis)

def amax(other, axis):
    f = lambda a, axis: np.amax(a, axis, keepdims=True)
    return _dim_npfunc_(f, other, axis)

def mean(other, axis):
    f = lambda a, axis: np.mean(a, axis, keepdims=True)
    return _dim_npfunc_(f, other, axis)
    
def stdv(other, axis):
    f = lambda a, axis: np.std(a, axis, keepdims=True)
    return _dim_npfunc_(f, other, axis)

def absolute(other):
    f = lambda a: np.absolute(a)
    return _npfunc_(f, other)

def capture_stats(data, var_norm, var_stan):
    stacked_data = data.exp_merge()
    stats = deepcopy(DataSet())
    stats[var_norm, 'mi'] = amin(stacked_data[var_norm, 0], axis=0)
    stats[var_norm, 'mx'] = amax(stacked_data[var_norm, 0], axis=0)
    stats[var_stan, 'me'] = mean(stacked_data[var_stan, 0], axis=0)
    stats[var_stan, 'sd'] = stdv(stacked_data[var_stan, 0], axis=0)
    return stats

def norm(data, stats):
    return (data - stats[:, 'mi']) / (stats[:, 'mx'] - stats[:, 'mi'])

def inv_norm(data, stats):
    return data * (stats[:, 'mx'] - stats[:, 'mi']) + stats[:, 'mi']

def stan(data, stats):
    return (data - stats[:, 'me']) / stats[:, 'sd']

def inv_stan(data, stats):
    return data * stats[:, 'sd'] + stats[:, 'me']

def rescale(data, stats):
    data = norm(data, stats)
    scaled_data = stan(data, stats)
    return scaled_data

def invscale(scaled_data, stats):
    scaled_data = inv_norm(scaled_data, stats)
    data = inv_stan(scaled_data, stats)
    return data

def augment(data, var_aug, n_rep, percent):
    d0, d1, d2, d3 = data.shape()
    noise = zeros((d0, d1, d2, n_rep)) | normal((var_aug, d1, d2, n_rep))
    aug_data = data.rep_stack(data * (1 + noise * percent))
    return aug_data
        
        
        
        
        
        
        

        

        

        

        


