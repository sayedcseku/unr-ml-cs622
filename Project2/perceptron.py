# -*- coding: utf-8 -*-
"""
@author: Md Abu Sayed
"""
import numpy as np

def run_epoch(X,Y,w,b):
    
    for i in range(len(X)):
        a = np.sum(w * X[i]) + b

        if(Y[i]*a <= 0):
            w = w + Y[i] * X[i]
            b = b + Y[i]
        #print("a,w,b: ",a, w, b)
    return w,b

def perceptron_train(X,Y):
    w = np.zeros(X.shape[1])
    b = 0.0
    
    temp_w, temp_b = run_epoch(X,Y,w,b)
    n_epoch = 1
    
    while( n_epoch <= 50 ):
        #print("Epoch: ", n_epoch)
        w,b = run_epoch(X,Y,w,b)     
        
        n_epoch += 1
    return w,b

def perceptron_test(X_test, Y_test, w, b):
    a = np.sum(X_test * w, axis = 1) + b
    y = np.where(a>0, 1,-1)
    acc = (y == Y_test).sum()/len(Y_test)
    return acc