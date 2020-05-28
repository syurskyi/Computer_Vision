# -*- coding: utf-8 -*-
"""
@author: abhilash
"""
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

class ShallowNetClass:
    def buildShallowNet(width, height, channels, classes):
        #initialize the model
        model = Sequential()
        #initialize the input shape as channels last
        imageInputShape = (height, width, channels)
        
        #Conv layer with 32 filters each with 3x3 size
        model.add(Conv2D(32, (3,3), padding="same", input_shape=imageInputShape))
        #add RELU activation fn to the Conv layer
        model.add(Activation("relu"))
        
        #Fully connected layer with softmax activation
        # flatten output from conv layer to an 1dimension array
        model.add(Flatten())
        #create a dense layer
        model.add(Dense(classes))
        #add softmax activation fn to the fully connected layer
        model.add(Activation("softmax"))
        
        #return the model
        return model
        

