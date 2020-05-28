# -*- coding: utf-8 -*-
"""
@author: abhilash
"""

#import the required libraries and classes
import numpy as np
from backpropagation import BackPropagation

#create the XOR dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

#initialize the back propagation multi layer perceptron class
bp_multi_layer_perceptron = BackPropagation([2,2,1], learning_rate=0.5)

#train the multi layer perceptron using train_fit method
print("Training the multi layer perceptron for XOR")
bp_multi_layer_perceptron.train_fit(X, y, no_of_epochs=1000)

#evaluate the prediction done by the perceptron
print("Starting the evaluation")

#loop through the dataset
for(input_data, actual_output) in zip(X, y):
    # get the predicted output
    sys_prediction = bp_multi_layer_perceptron.predict_eval(input_data)
    # define step function to convert all values > 0.5 to 1 and <0.5 to 0
    step_prediction_value = 1 if sys_prediction > 0.5 else 0
    #display the results
    print("for input {}, prediction is {}, step output is {}, ground truth is {}".format(input_data, sys_prediction, step_prediction_value, int(actual_output)))














