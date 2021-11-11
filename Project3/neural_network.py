# -*- coding: utf-8 -*-
import numpy as np

def calculate_loss(model, X,y):
    
    return loss

def predict(model, x):
    
    return y_pred


def build_model(X,y,nn_hdim, num_passes=20000, print_loss=False):
    
    nn_hdim = 500
    W1 = np.random.randn(nn_hdim, 2) 
    b1 = np.zeros((nn_hdim,1))
    
    W2 = np.random.randn(2, nn_hdim)
    b2 = np.zeros((2,1))
     
    a = np.dot(W1,X.T) + b1
    
    #implementing tanh activation function
    h = (np.exp(a)-np.exp(-a))/(np.exp(a)+np.exp(-a))
    
    z = np.dot(W2,h) + b2
    
    y_pred = np.exp(z)/np.sum(np.exp(z))
    
    return model