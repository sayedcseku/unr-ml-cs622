import numpy as np
from collections import Counter


class Node:
    
    def __init__(self, feature=None, threshold=None, left_node=None, right_node=None, gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left_node = left_node
        self.right_node = right_node
        self.gain = gain
        self.value = value
        
class DecisionTree:
    
    def __init__(self, min_samples_split=2, max_depth=5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        
    @staticmethod
    def _entropy(col):
        values, counts = np.unique(col, return_counts = True)
        entropy = 0
        for i in range(values.size):
            prob_i = counts[i]/np.sum(counts)
            entropy += ( prob_i * np.log2 (prob_i))
            
        return -1 * entropy
    
    def _information_gain(self, parent, left_child, right_child):
        
        num_left = len(left_child) / len(parent)
        num_right = len(right_child) / len(parent)
        
        # One-liner which implements the previously discussed formula
        return self._entropy(parent) - (num_left * self._entropy(left_child) + num_right * self._entropy(right_child))
    
    def _best_split(self, X, y):
       
        best_split = {}
        best_info_gain = -1
        n_rows, n_cols = X.shape
        
        # For every dataset feature
        for f_idx in range(n_cols):
            X_curr = X[:, f_idx]
            # For every unique value of that feature
            for threshold in np.unique(X_curr):
                # Construct a dataset and split it to the left and right parts
                # Left part includes records lower or equal to the threshold
                # Right part includes records higher than the threshold
                df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
                df_left = np.array([row for row in df if row[f_idx] <= threshold])
                df_right = np.array([row for row in df if row[f_idx] > threshold])

                # Do the calculation only if there's data in both subsets
                if len(df_left) > 0 and len(df_right) > 0:
                    # Obtain the value of the target variable for subsets
                    y = df[:, -1]
                    y_left = df_left[:, -1]
                    y_right = df_right[:, -1]

                    # Caclulate the information gain and save the split parameters
                    # if the current split if better then the previous best
                    gain = self._information_gain(y, y_left, y_right)
                    if gain > best_info_gain:
                        best_split = {
                            'feature_index': f_idx,
                            'threshold': threshold,
                            'df_left': df_left,
                            'df_right': df_right,
                            'gain': gain
                        }
                        best_info_gain = gain
        return best_split
    
    def _build(self, X, y, depth=0):
        
        n_rows, n_cols = X.shape
        
        # Check to see if a node should be leaf node
        if n_rows >= self.min_samples_split and depth <= self.max_depth:
            # Get the best split
            best = self._best_split(X, y)
            # If the split isn't pure
            if best['gain'] > 0:
                # Build a tree on the left
                left = self._build(
                    X=best['df_left'][:, :-1], 
                    y=best['df_left'][:, -1], 
                    depth=depth + 1
                )
                right = self._build(
                    X=best['df_right'][:, :-1], 
                    y=best['df_right'][:, -1], 
                    depth=depth + 1
                )
                return Node(
                    feature=best['feature_index'], 
                    threshold=best['threshold'], 
                    left_node=left, 
                    right_node=right, 
                    gain=best['gain']
                )
        # Leaf node - value is the most common target value 
        return Node(
            value=Counter(y).most_common(1)[0][0]
        )
    
    def fit(self, X, y):
       
        # Call a recursive function to build the tree
        self.root = self._build(X, y)
        
    def _predict(self, x, tree):
        
        # Leaf node
        if tree.value != None:
            return tree.value
        feature_value = x[tree.feature]
        
        # Go to the left
        if feature_value <= tree.threshold:
            return self._predict(x=x, tree=tree.left_node)
        
        # Go to the right
        if feature_value > tree.threshold:
            return self._predict(x=x, tree=tree.right_node)
        
    def predict(self, X):
       
        # Call the _predict() function for every observation
        return [self._predict(x, self.root) for x in X]
    
def DT_train_binary(X,Y,max_depth):
    
    left_Child_X = []
    left_Child_Y= []
    right_Child_X = []
    right_Child_Y = []
    
    if len(X) == 0:
        return;
        
    min_samples_split=2
    model = DecisionTree(min_samples_split,max_depth)
    model.fit(X, Y)
    
    return model    

def DT_test_binary(X,Y,DT):
    preds = DT.predict(X)
    Y = np.float32(Y)
    count = 0
    for i in range(len(Y)):
        if(preds[i] == Y[i]):
            count+=1
    
    accuracy = count/len(Y)
    return accuracy
