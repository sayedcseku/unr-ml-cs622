# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial import distance

def K_Means(X,K,mu):
    mu_updated = []
    iter = 1
    while(iter == 1 or not np.array_equal(mu,mu_updated)):
        clusters = [[] for i in range(K)]
        if (mu == []):
            #print("mu empty")
            #mu = mu.astype(np.float32)
            # for K random indices from training data X
            index = np.random.choice(X.shape[0], K, replace=False)  
            for i in range(len(index)):
                #print(X[index[i]])
                mu.append(X[index[i]])
            
            #converting the cluster centers list to array 
            mu = np.asarray(mu)
            
        else:
            mu = mu.astype(np.float32)

            if iter > 1:
                mu = mu_updated 
                mu_updated = []
        #print(mu,mu_updated)   
        
        #calculating distance of samples from cluster centers, mu and adding the sample to the minimum distant cluster
        dist = np.empty(K)
        for x in range(len(X)):
            for i in range(K):
                dist[i] = distance.euclidean(X[x],mu[i])
                
            min_index = np.argmin(dist)
            
            clusters[min_index].append(X[x])
        
        # Updating the cluster centers, mu
        
        for i in range(K):
            a = np.array(clusters[i])
            mu_updated.append( np.mean(a, axis=0))
        
        mu_updated = np.array(mu_updated)
        
        iter+=1
    
    return mu_updated


def K_Means_better(X,K):

    count = 0
    mu_set = []
    best_mu = []
    
    # appending array of cluster centers, mu in a list and keep track of highest appearing 'mu'
    # if any 'mu' appears more than three times, we are selecting that one as the best 'mu'
    
    while (count <= 2):
        mu = []
        mu_i = K_Means(X,K,mu)
        mu_set.append(mu_i)
        c = 0
        for i in range(len(mu_set)):
            if np.array_equal(mu_set[i],mu_i):
                c += 1
        if c> count:
            count = c
            best_mu = mu_i
            
    return best_mu