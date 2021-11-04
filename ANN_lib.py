# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 09:30:45 2020

@author: Alex Rogers
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from copy import deepcopy

class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x



class MSELoss(object):
    def __init__(self):
        return None
    
    def __call__(self, output, target, transfer_linears=None):
        loss = torch.mean((output - target) ** 2)
        return loss

class minDevMSELoss(object):
    def __init__(self, source_linears, memory):
        self.source_linears = source_linears
        self.memory = memory

    def __call__(self, output, target, transfer_linears):
        penalty = 0; n = 0
        for (source_layer, transfer_layer) in zip(self.source_linears, transfer_linears):
            penalty += torch.mean((source_layer.weight - transfer_layer.weight) ** 2)
            penalty += torch.mean((source_layer.bias - transfer_layer.bias) ** 2)
            n += 2
        penalty /= n
        MSE = torch.mean((output - target) ** 2)
        loss = MSE + penalty * self.memory
        return loss
    
class minDevMSELossCS2TM3(object):
    def __init__(self, source_linears, memory):
        self.source_linears = source_linears
        self.memory = memory

    def __call__(self, output, target, transfer_linears):
        penalty = 0; n = 0
        penalty += torch.mean((self.source_linears[0].weight - transfer_linears[0].weight[:, :-1]) ** 2)
        penalty += torch.mean((self.source_linears[0].bias - transfer_linears[0].bias) ** 2)
        for (source_layer, transfer_layer) in zip(self.source_linears[1:], transfer_linears[1:]):
            penalty += torch.mean((source_layer.weight - transfer_layer.weight) ** 2)
            penalty += torch.mean((source_layer.bias - transfer_layer.bias) ** 2)
            n += 2
        penalty /= n
        MSE = torch.mean((output - target) ** 2)
        loss = MSE + penalty * self.memory
        return loss


class transferFuncCS2TM3(object):
    def __init__(self, source_linears, memory):
        self.source_linears = source_linears
        self.memory = memory

    def __call__(self, output, target, transfer_linears):
        penalty = 0; n = 0
        penalty += torch.mean((self.source_linears[0].weight - transfer_linears[0].weight[:, :-1]) ** 2)
        penalty += torch.mean((self.source_linears[0].bias - transfer_linears[0].bias) ** 2)
        n += 2
        for (source_layer, transfer_layer) in zip(self.source_linears[1:], transfer_linears[1:]):
            penalty += torch.mean((source_layer.weight - transfer_layer.weight) ** 2)
            penalty += torch.mean((source_layer.bias - transfer_layer.bias) ** 2)
            n += 2
        penalty /= n
        MSE = torch.mean((output - target) ** 2)
        x = MSE / (penalty * self.memory)
        return x, penalty


class minDevMSELossCS2TM1(object):
    def __init__(self, source_linears, memory):
        self.source_linears = source_linears
        self.memory = memory

    def __call__(self, output, target, transfer_linears):
        penalty = 0; n = 0
        for (source_layer, transfer_layer) in zip(self.source_linears[:-1], transfer_linears[:-1]):
            penalty += torch.mean((source_layer.weight - transfer_layer.weight) ** 2)
            penalty += torch.mean((source_layer.bias - transfer_layer.bias) ** 2)
            n += 2
        penalty += torch.mean((self.source_linears[-1].weight - transfer_linears[-1].weight[:, :-1]) ** 2)
        penalty += torch.mean((self.source_linears[-1].bias - transfer_linears[-1].bias) ** 2)
        n += 2
        penalty /= n
        MSE = torch.mean((output - target) ** 2)
        loss = MSE + penalty * self.memory
        return loss


class transferFuncCS2TM1(object):
    def __init__(self, source_linears, memory):
        self.source_linears = source_linears
        self.memory = memory

    def __call__(self, output, target, transfer_linears):
        penalty = 0; n = 0
        for (source_layer, transfer_layer) in zip(self.source_linears[:-1], transfer_linears[:-1]):
            penalty += torch.mean((source_layer.weight - transfer_layer.weight) ** 2)
            penalty += torch.mean((source_layer.bias - transfer_layer.bias) ** 2)
            n += 2
        penalty += torch.mean((self.source_linears[-1].weight - transfer_linears[-1].weight[:, :-1]) ** 2)
        penalty += torch.mean((self.source_linears[-1].bias - transfer_linears[-1].bias) ** 2)
        n += 2
        penalty /= n
        MSE = torch.mean((output - target) ** 2)
        x = MSE / (penalty * self.memory)
        return x, penalty
    

class transferFunc(object):
    def __init__(self, source_linears, memory):
        self.source_linears = source_linears
        self.memory = memory

    def __call__(self, output, target, transfer_linears):
        penalty = 0; n = 0
        for (source_layer, transfer_layer) in zip(self.source_linears, transfer_linears):
            penalty += torch.mean((source_layer.weight - transfer_layer.weight) ** 2)
            penalty += torch.mean((source_layer.bias - transfer_layer.bias) ** 2)
            n += 2
        penalty /= n
        MSE = torch.mean((output - target) ** 2)
        x = MSE / (penalty * self.memory)
        return x, penalty




class SourceNet(nn.Module):
    def __init__(self, hyprams):
        super().__init__()
        
        # Defining ANN Topology:
        self.input_size    = hyprams['input_size']
        self.hl_sizes      = hyprams['hl_sizes']
        self.output_size   = hyprams['output_size']
        self.current_sizes = [self.input_size] + list(self.hl_sizes) + [self.output_size]
                
        # Defining Activation Functions:
        self.activations = nn.ModuleList()
        for activation in hyprams['activations']:
            if activation == 'Sigmoid':
                new_activation = nn.Sigmoid()
            elif activation == 'Tanh':
                new_activation = nn.Tanh()
            elif activation == 'LeakyReLU':
                new_activation = nn.LeakyReLU()
            elif activation == 'ReLU':
                new_activation = nn.ReLU()
            elif activation == 'Linear':
                new_activation = Linear()
            self.activations.append(new_activation)
            
        # Define Hidden Layers:
        self.linears = nn.ModuleList([nn.Linear(self.input_size, self.hl_sizes[0])])
        self.linears.extend([nn.Linear(self.hl_sizes[i], self.hl_sizes[i + 1]) for i in range(len(self.hl_sizes) - 1)])
        
        # Define Output Layer:
        self.linears.append(nn.Linear(self.hl_sizes[-1], self.output_size))

        
    def forward(self, x):   
        # Forward Pass Through All Layers + Activation Functions:
        for i, layer in enumerate(self.linears):
            x = layer(x)
            x = self.activations[i](x)
        return x
    
    def resetSource(self):
        return
    
    def initParams(self):
        for layer in self.linears:
            torch.nn.init.xavier_normal_(layer.weight.data)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

class SourceNetDropOut(nn.Module):
    def __init__(self, hyprams):
        super().__init__()
        
        # Defining ANN Topology:
        self.input_size    = hyprams['input_size']
        self.hl_sizes      = hyprams['hl_sizes']
        self.output_size   = hyprams['output_size']
        self.current_sizes = [self.input_size] + list(self.hl_sizes) + [self.output_size]
                
        # Defining Activation Functions:
        self.activations = nn.ModuleList()
        for activation in hyprams['activations']:
            if activation == 'Sigmoid':
                new_activation = nn.Sigmoid()
            elif activation == 'Tanh':
                new_activation = nn.Tanh()
            elif activation == 'LeakyReLU':
                new_activation = nn.LeakyReLU()
            elif activation == 'ReLU':
                new_activation = nn.ReLU()
            elif activation == 'Linear':
                new_activation = Linear()
            self.activations.append(new_activation)
            
        # Define Hidden Layers:
        self.linears = nn.ModuleList([nn.Linear(self.input_size, self.hl_sizes[0])])
        self.linears.extend([nn.Linear(self.hl_sizes[i], self.hl_sizes[i + 1]) for i in range(len(self.hl_sizes) - 1)])
        
        # Define Output Layer:
        self.linears.append(nn.Linear(self.hl_sizes[-1], self.output_size))
        
        # Dropout:
        self.dropout = [nn.Dropout(0.1), nn.Dropout(0)]
        
    def forward(self, x):   
        # Forward Pass Through All Layers + Activation Functions:
        for i, layer in enumerate(self.linears[:-1]):
            x = layer(x)
            x = self.dropout[i](x)
            x = self.activations[i](x)
        x = self.linears[-1](x)
        x = self.activations[-1](x)
        return x
    
    def resetSource(self):
        return
    
    def initParams(self):
        for layer in self.linears:
            torch.nn.init.xavier_normal_(layer.weight.data)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)


class TransferNet(SourceNet, nn.Module):
    def __init__(self, hyp):   
        super().__init__(hyp)
        # Load Source Model: (Assuming Source and Previous Model are Equal for Now)
        self.source_path = hyp['source_path']
        previous_model = torch.load(self.source_path)
        
        self.previous_sizes = self.source_sizes = previous_model.current_sizes
        
        with torch.no_grad():
            self.previous_biases = [layer.bias for layer in previous_model.linears]
            
            f_zeros = lambda i: torch.zeros((self.current_sizes[i + 1] - self.previous_sizes[i + 1], self.previous_sizes[i]))
            g_zeros = lambda i: torch.zeros((self.previous_sizes[i + 1] - self.source_sizes[i + 1], 1))
            h_zeros = lambda i: torch.zeros((self.current_sizes[i + 1] - self.previous_sizes[i + 1]))
            
            self.previous_weights = []; self.current_feedout = []; self.current_biases = [];
            
            for i, layer in enumerate(previous_model.linears):
                self.previous_weights.append(torch.cat((layer.weight, f_zeros(i))))
                self.current_feedout.append(g_zeros(i))
                self.current_biases.append(h_zeros(i))
    
    
    def resetSource(self):
        # - Overwrites With Source Weights and Biases - #
        
        # Resets Weights in Previous Model:
        for layer, weights, size in zip(self.linears, self.previous_weights, self.previous_sizes):
            layer.weight.data[:, :size] = weights
            
        # Resets Biases in Previous Model:
        for layer, biases, size in zip(self.linears, self.previous_biases, self.previous_sizes[1:]):
            layer.bias.data[:size] = biases
            

        # --- Enforces Implicit Physical Assumptions --- #
        
        # Zeros Connections Between Source Model Input Nodes and First Hidden Layer Transfer Nodes:
        for i, layer in enumerate(self.linears[:-1]):
            layer.weight.data[self.source_sizes[i + 1]:self.previous_sizes[i + 1], self.previous_sizes[i]:] = self.current_feedout[i]

        # Zeros Biases in Transfer Nodes:
        for layer, zeros, size in zip(self.linears[:-1], self.current_biases[:-1], self.previous_sizes[1:]):
            layer.bias.data[size:] = zeros



class PartTruncatedSourceNet(SourceNet, nn.Module):
    def __init__(self, hyp):   
        super().__init__(hyp)
        self.source_model = torch.load(hyp['source_path']) 

    def resetSource(self):
        self.linears[1].weight = self.source_model.linears[1].weight
        self.linears[1].bias = self.source_model.linears[1].bias
        return



class TruncatedTransferNetTM1(SourceNet, nn.Module):
    def __init__(self, hyp):   
        super().__init__(hyp)
        self.source_model = torch.load(hyp['source_path']) 
        
        self.source_model.linears[0].weight.requires_grad = False
        self.source_model.linears[1].weight.requires_grad = False
        self.source_model.linears[0].bias.requires_grad = False
        self.source_model.linears[1].bias.requires_grad = False

    def resetSource(self):
        self.linears[0].weight = self.source_model.linears[0].weight
        self.linears[0].bias = self.source_model.linears[0].bias
        self.linears[1].weight = self.source_model.linears[1].weight
        self.linears[1].bias = self.source_model.linears[1].bias
        return


class TruncatedTransferNetTM2(SourceNet, nn.Module):
    def __init__(self, hyp):   
        super().__init__(hyp)
        self.source_model = torch.load(hyp['source_path']) 
        
        self.source_model.linears[0].weight.requires_grad = False
        self.source_model.linears[0].bias.requires_grad = False

    def resetSource(self):
        self.linears[0].weight = self.source_model.linears[0].weight
        self.linears[0].bias = self.source_model.linears[0].bias
        return
    


## ========================================================================= ##
# -------------------------- SalicylicTransferNet --------------------------- #
## ========================================================================= ##

class SalicylicTransferNetTM1(nn.Module):
    def __init__(self, hyp):
        super().__init__()
        
        self.source_path = hyp['source_path']
        source_model = torch.load(self.source_path)
        self.output_size = hyp['output_size']
        self.modifier = nn.Linear(1, self.output_size)
        self.source_size = source_model.current_sizes

        self.activations = source_model.activations
        self.source_linears = source_model.linears
        
        self.source_linears[0].weight.requires_grad = False
        self.source_linears[1].weight.requires_grad = False
        
        self.source_linears[0].bias.requires_grad = False
        self.source_linears[1].bias.requires_grad = False
        
        self.modifier = nn.Linear(self.source_size[-2] + 1, self.output_size)
        self.linears = nn.ModuleList([self.source_linears[0],
                                     self.modifier])
        
    def forward(self, x):   
        # Forward Pass Through All Layers + Activation Functions:
        xs = x[:, :-1]
        xt = x[:, -1:]
        for i, layer in enumerate(self.linears[:-1]):
            xs = layer(xs)
            xs = self.activations[i](xs)
        x = torch.cat([xs, xt], dim=1)
        x = self.linears[-1](x)
        x = self.activations[-1](x)
        return x
    
    def initParams(self):
        torch.nn.init.xavier_normal_(self.modifier.weight.data)
        return
    
    def resetSource(self):
        self.linears[-1].weight.data[:, :-1] = self.source_linears[1].weight.data
        self.linears[-1].bias.data = self.source_linears[1].bias.data
        return
        
class SalicylicTransferNetTM2(nn.Module):
    def __init__(self, hyp):
        super().__init__()
        
        self.source_path = hyp['source_path']
        source_model = torch.load(self.source_path)
        self.output_size = hyp['output_size']
        self.modifier = nn.Linear(1, self.output_size)
        self.source_size = source_model.current_sizes

        self.activations = source_model.activations
        self.source_linears = source_model.linears
        
        self.source_linears[0].weight.requires_grad = False
        
        self.source_linears[0].bias.requires_grad = False
        
        self.modifier = nn.Linear(self.source_size[-2] + 1, self.output_size)
        self.linears = nn.ModuleList([self.source_linears[0],
                                     self.modifier])
        
    def forward(self, x):   
        # Forward Pass Through All Layers + Activation Functions:
        xs = x[:, :-1]
        xt = x[:, -1:]
        for i, layer in enumerate(self.linears[:-1]):
            xs = layer(xs)
            xs = self.activations[i](xs)
        x = torch.cat([xs, xt], dim=1)
        x = self.linears[-1](x)
        x = self.activations[-1](x)
        return x
    
    def initParams(self):
        torch.nn.init.xavier_normal_(self.modifier.weight.data)
        return
    
    def resetSource(self):
        return
        



class SalicylicTransferNetTM3(SourceNet, nn.Module):
    def __init__(self, hyp):   
        super().__init__(hyp)
        # Load Source Model: (Assuming Source and Previous Model are Equal for Now)
        self.source_path = hyp['source_path']
        previous_model = torch.load(self.source_path)
        
        self.previous_sizes = self.source_sizes = previous_model.current_sizes

        
        with torch.no_grad():
            self.previous_biases = [layer.bias for layer in previous_model.linears]
            
            f_zeros = lambda i: torch.zeros((self.current_sizes[i + 1] - self.previous_sizes[i + 1], self.previous_sizes[i]))
            h_zeros = lambda i: torch.zeros((self.current_sizes[i + 1] - self.previous_sizes[i + 1]))
            
            self.previous_weights = []; self.current_feedout = []; self.current_biases = [];
            
            for i, layer in enumerate(previous_model.linears):
                self.previous_weights.append(torch.cat((layer.weight, f_zeros(i))))
                self.current_biases.append(h_zeros(i))
    
    
    def resetSource(self):
        # - Overwrites With Source Weights and Biases - #
        # Resets Weights in Previous Model:
        for layer, weights, size in zip(self.linears, self.previous_weights, self.previous_sizes):
            layer.weight.data[:, :size] = weights
            
        # Resets Biases in Previous Model:
        for layer, biases, size in zip(self.linears, self.previous_biases, self.previous_sizes[1:]):
            layer.bias.data[:size] = biases
            
            
class SalicylicTransferNetTM5(nn.Module):
    def __init__(self, hyp):
        super().__init__()
        
        self.source_path = hyp['source_path']
        source_model = torch.load(self.source_path)
        self.output_size = hyp['output_size']
        self.modifier = nn.Linear(1, self.output_size)
        self.source_size = source_model.current_sizes

        self.activations = source_model.activations
        self.source_linears = source_model.linears
        
        self.modifier = nn.Linear(self.source_size[-2] + 1, self.output_size)
        self.linears = nn.ModuleList([self.source_linears[0],
                                     self.modifier])
        
    def forward(self, x):   
        # Forward Pass Through All Layers + Activation Functions:
        xs = x[:, :-1]
        xt = x[:, -1:]
        for i, layer in enumerate(self.linears[:-1]):
            xs = layer(xs)
            xs = self.activations[i](xs)
        x = torch.cat([xs, xt], dim=1)
        x = self.linears[-1](x)
        x = self.activations[-1](x)
        return x
    
    def initParams(self):
        torch.nn.init.xavier_normal_(self.modifier.weight.data)
        return         
            
    def resetSource(self):
        return
    
    
    
    
    