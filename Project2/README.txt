# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 04:37:11 2021

@author: Md Abu Sayed
"""

1. Perceptron
=============

# perceptron_train(X,Y):

This function is used to train the perceptron algorithm. It utlizes the run_epoch(X,Y,w,b) algorithm.
The algorithm is build on following steps:
    1. Initilize the weight, w and bias term, b to 0. Size of w depends on the size of X (features i = 1 to D)
    2. Compute the activation, a for training samples:
        a = np.sum(w * X[i]) + b, i is each training samples
    3. If Y*a <= 0, update the w and b with the update rules
    4. Repeat 2 and 3 for all training samples
    
    5. Repeat 2-4 (which is in the run_epoch() function) untill convergence or fixed number of epochs. In this code we used 50 epochs


# perceptron_test(X_test, Y_test, w, b):

Steps:
    1. Compute the activation function a.
    2. Check if a>0, then it's classfied as +1, otherwise -1.
    3. Compute accuracy and return it.
    

2. Gradient Descent
===================

# gradient_descent(grad_f, x_init, eta):

Steps:
    1. It takes the derivative of the loss function, initial X and eta as input
    2. It computes the gradient with respect to X and the previous function.
    3. It updates the X with X = X - eta * grad_X equation
    4. It continues to update the X untill magnitude of grad_X/ l2 norm of the grad_X is less than 0.0001
        

