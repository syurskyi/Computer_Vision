# -*- coding: utf-8 -*-
"""

@author: abhilash
"""

#importing libraries
import numpy as np

class SimplePerceptron:
    def __init__(self, no_of_inputs, learning_rate=0.1):
        #generate a random value weight matrix
        self.W = np.random.randn(no_of_inputs+1) / np.sqrt(no_of_inputs)
        #set the learning rate
        self.learning_rate = learning_rate
        
    #defining the step function
    def step(self, input_x):
        #the logic of step activation
        return 1 if input_x >0 else 0
    
    #defining the train_fit method
    def train_fit(self, X, y, no_of_epochs=10):
        #add a one column to the input feature matrix
        X = np.c_[X, np.ones((X.shape[0]))]
        
        #loop through every epoch
        for single_epoch in np.arange(0, no_of_epochs):
            #loop through every data point
            for (training_input, expected_output) in zip(X,y):
                #find prediction, dot product of W and input feature matrix 
                prediction = np.dot(training_input,self.W)
                # pass the prediction to step function and get the actual prediction output
                prediction = self.step(prediction)
                
                #if prediction is not equal to the expected output, update weight
                if prediction != expected_output:
                    #find the error value
                    error_value = prediction - expected_output
                    #update the weight matrix
                    self.W += -self.learning_rate * error_value * training_input
    
    #define the predict_evaluation method
    def predict_eval(self, X):
        #convert to 2d matrix if its not 2d
        X = np.atleast_2d(X)
        #add a one column to the input feature matrix
        X = np.c_[X, np.ones((X.shape[0]))]        
        #find the prediction and return it
        return self.step(np.dot(X,self.W))
    
    
    
