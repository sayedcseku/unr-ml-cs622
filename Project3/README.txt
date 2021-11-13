Project 3: Neural Network (2 Layers)
====================================


- I have created a 2-layer neural network. The network takes 2D data as input, that is x1 and x2.
The output of the network will be 2D as well, that is y_pred is a 2D vector. If x is the 2-dimensional input to our
network then we calculate our prediction y_pred (also two-dimensional) as follows:

a = x . W1 + b1
h = tanh(a)
z = h . W2 + b2
y_pred = softmax(z)

- Learning the parameters for our network means finding parameters (W1, b1, W2, b2) that minimize the
error on our training data. The softmax output we used, is the categorical cross-entropy loss function
(also known as negative log likelihood). 

- All of the applied formula are described in the problem description PDF. I also added comments on code part that might be unexplained.
Most of the codes are self explantory when compared with the forward propagation and backpropagation rules. 


- The mathematical explanation about the softmax function is added as photo.