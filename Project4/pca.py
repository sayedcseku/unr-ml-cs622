import numpy as np

def compute_Z(X, centering=True, scaling=False):
    
    Z = X
    
    if centering:
        X_mean = np.mean(X, axis=0)
        #X_mean = X_mean.reshape(X_mean.shape[0],1)
        #Z = np.subtract(Z,X_mean)
        Z = Z - X_mean
    
    if scaling:
        X_std = np.std(X,axis=0)
        Z = Z/X_std
    
    return Z

def compute_covariance_matrix(Z):
    cov_Z = np.dot(Z.T,Z)
    return cov_Z

def find_pcs(COV):
    L, pcs =  np.linalg.eig(COV)
    index = L.argsort()[::-1]                                         #sort in desc order 
    return L[index], pcs[:,index]

def project_data(Z, PCS, L, k, var):
    if k>0:
        project_mat = PCS[:,:k]
    elif var>0:
        project_mat = PCS[:,:k]
    
    Z_star = np.dot(project_mat.T,Z.T).T
    return Z_star    