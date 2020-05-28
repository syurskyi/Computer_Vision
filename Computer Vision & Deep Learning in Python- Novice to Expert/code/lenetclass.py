# -*- coding: utf-8 -*-
"""
@author: abhilash
"""
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

class LeNetClass:
    def buildLeNet(width, height, channels, classes):
        #initialize the model
        model = Sequential()
        #initialize the input shape as channels last
        imageInputShape = (height, width, channels)
        
        #First Conv layer with 20 filters each with 5x5 size
        model.add(Conv2D(20, (5,5), padding="same", input_shape=imageInputShape))
        #add RELU activation fn to the first Conv layer
        model.add(Activation("relu"))
        #First pooling layer with poolsize 2x2 , stride 2x2, using the maxpooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        
        #Second Conv layer with 50 filters each with 5x5 size
        model.add(Conv2D(50, (5,5), padding="same"))
        #add RELU activation fn to the first Conv layer
        model.add(Activation("relu"))
        #Second pooling layer with poolsize 2x2 , stride 2x2, using the maxpooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        
        
        # flatten output from conv layer to an 1dimension array
        model.add(Flatten())
        
        #create the first fully connected (dense) layer with 500 nodes
        model.add(Dense(500))
        #add relu activation fn to the first fully connected layer
        model.add(Activation("relu"))
        
        #create the next fully connected (dense) layer
        model.add(Dense(classes))
        #add softmax activation fn to the fully connected layer
        model.add(Activation("softmax"))
        
        #return the model
        return model
        

