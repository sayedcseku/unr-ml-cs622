1. Decision Tree:
=================

Disclaimer: 

> Used a reference tree implementation from internet, however, implemented the Entropy, Information Gain, etc. function according to my intuition
> Counter package is used to count maximum occurance of a label

DT TRAIN BINARY

> DT_train_binary(X,Y,max_depth) will create an model object with DecisionTree(min_samples_split,max_depth) call, for binary tree min_samples_split is 2
> then model.fit(X,Y) will train the decision tree with training data and label
    - the fit(X,Y) method will call the _build() method which builds the decision tree based on best['gain']. It's also used to decide which sample goes to the left 
        and which one goes as a right child/node
        - We created the best['gain'] with _best_split(X,y)
            - the _best_split() function utilizes _information_gain() function
            - _information_gain() function utilizes the _entropy(col) function

DT_test_binary(X,Y,DT):

> The function uses train DT model to predict Test sample Y.
> the prediction preds is then compared with labels Y to calculate accuracy  




2.  Nearest Neighbors:
======================
KNN_test(X_train,Y_train,X_test,Y_test,K) 

Parameters:
    X_train,Y_train -> Training samples and their labels in array
    X_test,Y_test   -> Test samples and their labels in array
    K               -> Number of nearest Neighbours to be compared

> A distance matrix dist is computed based on Euclidean distances between test samples and training samples
> K number of neighbours are selected based on minimum distance from dist matrix
> The sum of the labels of these selected neighbours are calculated. If sum > 0 then the positive class, else the test sample is a negative classes.
> Accuracy is calculated based on the accurate number of predictions.

CHOOSE K 

> Iterate the above function with odd K values, find the accuracy with respect to that K and store them in an array
> Return K with the highest accuracy




3. Clustering:
==============

K-MEANS CLUSTERING

Parameters:
    X -> Feature of data samples
    K -> Number of Clusters
    mu -> Given Cluster centers

Part 1: Where mu is defined
    
> 1. First compute distances from X to all mu, add the the sample to the closest cluster
> 2. Recompute the cluster centers, mu for 
> 3. Continue to do 1 and 2 until updated cluster centers are the same to the previous cluster centers 


Part 2: Where mu is not defined

> Check for if the mu passed as an input is an empty list
> If it is,  choose random K sample numbers from X with np.random.choice function and assign those samples as mu

KMEANS CLUSTERING AND GENERATING THE BEST MU , K_Means_better(X,K):

> Run the previous function with mu=[], and with different K values from 1 to n with 2 steps [odd numbers]
> For each K, save the cluster centers mu
> Select the best cluster center mu, if it appears more than 3 times

