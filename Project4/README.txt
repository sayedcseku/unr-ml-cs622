Project 4: PCA Implementation and Image Compression with PCA
============================================================

1. PCA:
=======

def compute_Z(X, centering=True, scaling=False):

- If centering is True, subtracted mean from X
- If scaling is True, X is divided the standard deviation 


def compute_covariance_matrix(Z):

- Takes the standardized data matrix Z and return the covariance matrix Z_T . Z=COV (anumpy array).


def find_pcs(COV):

- Finds the principal components and sort them in decreasing order

def project_data(Z, PCS, L, k, var):

- Projects the data into k principal components



2. Application in Image Compression:
====================================

def load_data(input_dir):

- It reads image data from a derectory with os package
- Then flattens each image, converts into floating point number, then adds them as a image column


def compress_images(DATA,k):

- It compresses the loaded images with PCA algorithm described in 1
- Creates a 'Output' direcrory if it doesn't' exists
- Save the compressed images in the Output directory with pyplot.imsave(), with cmap='gray'