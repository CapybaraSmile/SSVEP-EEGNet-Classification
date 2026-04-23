import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter，filtfilt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# EEGNet-specific imports
from EEGModels import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
import tensorflow.keras.models as models
import tensorflow.compat.v1 as tf
from tslearn.metrics import soft_dtw
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from sklearn.model_selection import KFold
from scipy import io, signal
import math

def build_fusion_model():
        global layers_i
        model1 = EEGNet(nb_classes = n_freqs, Chans = chans, Samples = samples, dropoutRate = 0.5, kernLength = 125, F1 = 8, D = 2, F2 = 16, 
               dropoutType = 'Dropout')

        for layer in model1.layers:
            layers_i = layers_i + 1
            layer.name = layer.name + str(layers_i)
    
        model2 = EEGNet(nb_classes = n_freqs, Chans = chans, Samples = samples, dropoutRate = 0.5, kernLength = 125, F1 = 8, D = 2, F2 = 16, 
               dropoutType = 'Dropout')

        for layer in model2.layers:
            layers_i = layers_i + 1
            layer.name = layer.name + str(layers_i)
            
        model3 = EEGNet(nb_classes = n_freqs, Chans = chans, Samples = samples, dropoutRate = 0.5, kernLength = 125, F1 = 8, D = 2, F2 = 16, 
               dropoutType = 'Dropout')

        for layer in model3.layers:
            layers_i = layers_i + 1
            layer.name = layer.name + str(layers_i)
            
        features = layers.concatenate([model1.layers[-2].output, model2.layers[-2].output, model3.layers[-2].output])
        features = layers.Dense(10, activation="softmax")(features)
        fusion_model = models.Model([model1.input, model2.input, model3.input], features)
        fusion_model.compile(loss='categorical_crossentropy', optimizer = "Adam", metrics = ["accuracy"])
        return fusion_model