# -*- coding: utf-8 -*-
"""
@author: abhilash
"""
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout


class VGGNetClass:
    def buildVGGNet(width, height, channels, classes):
        #initialize the model
        model = Sequential()
        #initialize the input shape as channels last
        imageInputShape = (height, width, channels)
        
        #First Conv layer with 32 filters each with 3x3 size
        model.add(Conv2D(32, (3,3), padding="same", input_shape=imageInputShape))
        #add RELU activation fn to the first Conv layer
        model.add(Activation("relu"))
        #add batch normalization to prevent internal covariate shift
        model.add(BatchNormalization(axis=-1))
        
        #Second Conv layer with 32 filters each with 3x3 size
        model.add(Conv2D(32, (3,3), padding="same"))
        #add RELU activation fn to the first Conv layer
        model.add(Activation("relu"))
        #add batch normalization to prevent internal covariate shift
        model.add(BatchNormalization(axis=-1))
        
        #First pooling layer with poolsize 2x2 
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        
        #Third Conv layer with 64 filters each with 3x3 size
        model.add(Conv2D(64, (3,3), padding="same"))
        #add RELU activation fn to the first Conv layer
        model.add(Activation("relu"))
        #add batch normalization to prevent internal covariate shift
        model.add(BatchNormalization(axis=-1))
        
        #Fourth Conv layer with 64 filters each with 3x3 size
        model.add(Conv2D(64, (3,3), padding="same"))
        #add RELU activation fn to the first Conv layer
        model.add(Activation("relu"))
        #add batch normalization to prevent internal covariate shift
        model.add(BatchNormalization(axis=-1))
        
        #Second pooling layer with poolsize 2x2 
        model.add(MaxPooling2D(pool_size=(2,2)))
        #dropout layer to prevent overfitting
        model.add(Dropout(0.25))

        
        # flatten output from conv layer to an 1dimension array
        model.add(Flatten())
        
        #create the first fully connected (dense) layer with 500 nodes
        model.add(Dense(512))
        #add relu activation fn to the first fully connected layer
        model.add(Activation("relu"))
        #add batch normalization to prevent internal covariate shift
        model.add(BatchNormalization(axis=-1))
        #dropout layer to prevent overfitting
        model.add(Dropout(0.5))
        
        #create the next fully connected (dense) layer
        model.add(Dense(classes))
        #add softmax activation fn to the fully connected layer
        model.add(Activation("softmax"))
        
        #return the model
        return model
        

