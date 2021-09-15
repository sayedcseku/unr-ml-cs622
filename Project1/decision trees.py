# -*- coding: utf-8 -*-
import numpy as np

def entropy(col):
    values, counts = np.unique(col, return_counts = True)
    entropy = 0
    for i in range(values.size):
        prob_i = counts[i]/np.sum(counts)
        entropy += ( prob_i * np.log2 (prob_i))
        
    return -1 * entropy

def infoGain (data, split_feature, target):
   
    #Entropy of total dataset
    H_S = entropy(data[target])   
    
    # Split on target feature and count values: counts
    values,counts= np.unique(data[split_feature],return_counts=True)
    
    #Calculate the weighted entropy, SUM(P(t) * H(t))
    P_t_H_t = 0
    for i in ranges(values.size):
        prob_t = counts[i]/np.sum(counts)
        P_t_H_t += prob_t * entropy()
    
def DT_train_binary(X,Y,max_depth):
    
    return model
    

def DT_test_binary(X,Y,DT):
    return accuracy