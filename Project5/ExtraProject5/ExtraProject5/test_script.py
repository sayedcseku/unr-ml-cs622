import numpy as np
import matplotlib.pyplot as plt
import helpers
import sys
#This should run outside of your Project5 Directory
sys.path.insert(0, 'Project5')
import svm

data = helpers.generate_training_data_binary(1)
[w,b,S] = svm.svm_train_brute(data)
print(w,b,S)

data = helpers.generate_training_data_binary(2)
[w,b,S] = svm.svm_train_brute(data)
print(w,b,S)

data = helpers.generate_training_data_binary(3)
[w,b,S] = svm.svm_train_brute(data)
print(w,b,S)

data = helpers.generate_training_data_binary(4)
[w,b,S] = svm.svm_train_brute(data)
print(w,b,S)


[data,Y] = helpers.generate_training_data_multi(1)
[W,B] = svm.svm_train_multiclass(data)
print(W,B)
