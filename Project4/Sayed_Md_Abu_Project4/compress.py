import pca as pca
import numpy as np
import matplotlib.pyplot as plt
import os

def compress_images(DATA,k):
    
    Z = pca.compute_Z(DATA)
    COV = pca.compute_covariance_matrix(Z)
    L, PCS = pca.find_pcs(COV)
    Z_star = pca.project_data(Z, PCS, L, k, 0)
    
    X_compressed = np.dot(Z_star,PCS[:,:k].T)
    
    # Directory
    directory = "Output"
      
    # Parent Directory path
    parent_dir = "Data"
      
    # Path
    path = os.path.join(parent_dir, directory)
      
    # Create the directory
    # 'Output' in
    # 'Data/Output /'
    isExist = os.path.exists(path)

    if not isExist:
      
      # Create a new directory because it does not exist 
      os.makedirs(path)
    
    for i in range(X_compressed.shape[1]):
        img1 = X_compressed[:,i].reshape((60,48))
        
        save_path = os.path.join(path, 'image_'+ str(i)+ '.png')
        #plt.imsave(save_path, img1, vmin=0, vmax=255, cmap='gray')
        plt.imsave(save_path, img1, cmap='gray')
    return


def load_data(input_dir):
    
    images = []
    for filename in os.listdir(input_dir):
        img = plt.imread(os.path.join(input_dir,filename))
        if img is not None:
            images.append(img.flatten())
    #img = plt.imread('Data/Train/00001_930831_fa_a.pgm')
    flatten_imgs = np.stack(images, axis=1)
    return flatten_imgs.astype(float)
