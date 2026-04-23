import numpy as np
import math


def get_train_test(train_indices, test_indices, segmented_data):
    # training and test set, adapt the format to input to the EEGNet
    X_train   = segmented_data[:, :, train_indices, :, :] 
    targets1  = X_train.shape[0]
    channels1 = X_train.shape[1]
    trials1   = X_train.shape[2]
    segments1 = X_train.shape[3]
    samples1  = X_train.shape[4]
    X_train   = X_train.transpose(3, 0, 2, 1, 4).reshape(segments1*targets1*trials1, channels1, samples1)
        
    X_test    = segmented_data[:, :, test_indices, :, :]
    targets2  = X_test.shape[0]
    channels2 = X_test.shape[1]
    trials2   = X_test.shape[2]
    segments2 = X_test.shape[3]
    samples2  = X_test.shape[4]
    X_test    = X_test.transpose(3, 0, 2, 1, 4).reshape(segments2*targets2*trials2, channels2, samples2)
        
    # generaate labels for training and test set
    Y_train = []
    for i in range(segments1):
        for j in range(targets1):
            for k in range(trials1):
                Y_train.append(j)
    Y_train = np_utils.to_categorical(Y_train)
        
    Y_test = []
    for i in range(segments2):
        for j in range(targets2):
            for k in range(trials2):
                Y_test.append(j)
    Y_test = np_utils.to_categorical(Y_test)
    
    return X_train, X_test, Y_train, Y_test