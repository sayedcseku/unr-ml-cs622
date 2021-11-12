# -*- coding: utf-8 -*-
import numpy as np

def softmax(x):
    
    max = np.max(x,axis=1,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.exp(x - max) #subtracts each row with its max value
    sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / sum 
    return f_x

def tanh(a):
    #implementing tanh activation function
    h = (np.exp(a)-np.exp(-a))/(np.exp(a)+np.exp(-a))
    return h

def calculate_loss(model, X,y):
    
    y_pred = predict(model, X)
    N = y.shape[0]
    
    loss = -1 * np.sum(y * np.log(y_pred)) / N # the BCE loss function
    return loss

def predict(model, X):
    
    a = np.dot(X,model['W1']) + model['b1']
    
    h = tanh(a)
    
    z = np.dot(h,model['W2']) + model['b2']
    
    y_pred = softmax(z)
    
    return y_pred




def build_model(X,y,nn_hdim, num_passes=20000, print_loss=False):
    
    nn_hdim = 500
    input_dim = 2
    output_dim = 2
    W1 = np.random.randn(input_dim,nn_hdim) 
    b1 = np.zeros((1,nn_hdim))
    
    W2 = np.random.randn(nn_hdim,output_dim)
    b2 = np.zeros((1,output_dim))
    
    y_2d = np.eye(2)[y] #converting 1d y to 2d y
    
    for i in range(num_passes):
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2':b2}
        
        # Forward Propagation
        
        a = np.dot(X,model['W1']) + model['b1']
    
        h = tanh(a)
    
        y_pred = predict(model, X)
        #loss = calculate_loss(model, X, y_2d)
        
        
        
        #Backpropagation
        #dy = y_pred - y_2d
        dy = y_pred
        
        for j in range(len(y_pred)):
            # if the label = 0 then y_hat - y apply for first output node
            if y[j] == 0:
                dy[j, 0] = y_pred[j, 0] - 1
                dy[j, 1] = y_pred[j, 1]
            # if the label = 1 then second output, y_hat - y
            else:
                dy[j, 0] = y_pred[j, 0]
                dy[j, 1] = y_pred[j, 1] - 1
        
        dA = (1-tanh(a)*tanh(a)) * np.dot(dy, W2.T)
        
        dW2 = np.dot(h.T,dy)
        
        db2 = np.sum(dy, axis=0)
        
        dW1 = np.dot(X.T, dA)
        
        db1 = np.sum(dA,axis=0)
        
        W1 = W1 - 0.01 * dW1
        b1 = b1 - 0.01 * db1
        
        W2 = W2 - 0.01 * dW2
        b2 = b2 - 0.01 * db2
        
        if print_loss and i % 1000 == 0:
            print("Iteration : ", i, "Loss : ", calculate_loss(model, X, y_2d))
        
    return model