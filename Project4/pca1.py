# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 00:17:06 2021

@author: HP
"""

import numpy as np
import sys


def compute_Z(arr, centering=True, scaling=False):
    rows, columns = arr.shape
    Z_mat = np.zeros(shape=(rows, columns))
    a=[]
    tempArray = np.zeros(rows)
   
    for column in range(columns):        
        mean = np.mean(arr[:,column])                               #computer mean
        std = np.std(arr[:,column])                                 #computer std deviation
        if (centering == True):
            for element in arr[:,column]:             
                tempArray = np.append(tempArray, (element - mean))  #append matrix elements
        elif (scaling == True):
            for element in arr[:,column]:
                tempArray = np.append(tempArray, (element / std))
                  
    tempArray= np.trim_zeros(tempArray, 'f')                        #drop the forward zeros
    Z_mat = np.reshape(tempArray,(columns,rows))                    #reshape into columns and return
    return Z_mat

def compute_covariance_matrix(Z):
    ZT=np.transpose(Z)                                              #find transpose
    cov = np.dot(Z,ZT)                                              #compute covariance matrix
    return cov
    
def find_pcs(COV):
    L, PCS = np.linalg.eig(COV)                                     #get eigen values and vectors
    idx = L.argsort()[::-1]                                         #sort in descending order 
    L = L[idx] 
    PCS = PCS[:,idx]                    
    return L, PCS

def project_data(Z, PCS, L, k, var):
    if k>0:
        projection_matrix=PCS.T[:,:k]                               #calculate projection matrix
    elif var>0:
        projection_matrix=PCS.T[:,:k]
   
    Z_star = np.dot(projection_matrix.T, Z).T                       #calculate Z_star
    return Z_star