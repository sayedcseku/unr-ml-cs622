# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial import distance

def KNN_test(X_train,Y_train,X_test,Y_test,K):
    
    n_train = len(X_train)
    n_test = len(X_test)
    
    dist = np.zeros(n_train)
    
    acc_pred = 0
    for i in range(n_test):
        for j in range(n_train):
            dist[j] = distance.euclidean(X_train[i],X_test[j])
            
        index_train = np.argsort(dist)
        sum=0
        for k in range(K):
            sum = sum +  Y_train[index_train[k]]
            #print((dist[index_train[k]]," Y: ",  Y_train[index_train[k]] , "SUM: ", sum 
    
        
        #print("SUM: ")        
        if sum > 0:
            y = 1
        else:
            y = -1
        
        if Y_test[i] == y:
            acc_pred += 1
                
    accuracy = acc_pred / n_test
    return accuracy

def choose_K(X_train,Y_train,X_val,Y_val):
    K = np.zeros(10)
    for i in range(1, 10, 2):
        K[i] = KNN_test(X_train,Y_train,X_val,Y_val,i)
    
    return K.argmax()