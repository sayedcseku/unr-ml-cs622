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
    weightedEntropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_feature]==values[i]).dropna()[target]) for i in range(len(values))]) 
    
    infoGain = H_S - weightedEntropy
    
    return infoGain
    
def DT_train_binary(X,Y,max_depth):
    
    return model
    

def DT_test_binary(X,Y,DT):
    return accuracy