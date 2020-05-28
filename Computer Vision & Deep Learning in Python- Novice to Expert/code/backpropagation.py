# -*- coding: utf-8 -*-
"""
@author: abhilash
"""
#importing the basic packages
import numpy as np

#declaring the class and the constructor
class BackPropagation:
    def __init__(self, layers_list, learning_rate=0.1):
        #have an empty list for weight matrix
        self.W = []
        #set the learning rate
        self.learning_rate = learning_rate
        #set the layers for the network
        self.layers_list = layers_list
        
        #for loop for generating the weight matrix for 
        #inner layers except the last two
        for i in np.arange(0, len(layers_list) - 2):
            #generate random value weight matrix
            w = np.random.randn(layers_list[i] + 1, layers_list[i + 1] + 1)
            self.W.append(w / np.sqrt(layers_list[i]))
        
        #generate the weight matrix for the last two layers
        w = np.random.randn(layers_list[-2] + 1, layers_list[-1])
        self.W.append(w / np.sqrt(layers_list[-2]))
        
    #define sigmoid activation function
    def sigmoid_activation_function(self, x):
        #calculating the sigmoid activation value
        return 1.0/(1+np.exp(-x)) 
        
    #define derivative of sigmoid activation 
    def deriv_sigmoid_activation_function(self, x):
        #calculating the derrivative
        return x * (1 - x)      
        
    #defining the train_fit method
    def train_fit(self, X, y, no_of_epochs=1000):
        #add a one column to the input feature matrix
        X = np.c_[X, np.ones((X.shape[0]))]
        
        #loop through every epoch
        for single_epoch in np.arange(0, no_of_epochs):
            #loop through every data point
            for (training_input, expected_output) in zip(X,y):
                #METHOD FOR PARTIAL FIT
                self.fit_fwd_bwd_partial(training_input, expected_output)
                #OPTION TO DISPLAY THE PROGRESS OF TRAINING
                #DISPLAY CURRENT ERROR STATUS 
                loss = self.find_loss(X, y)
                if(single_epoch==0 or (single_epoch+1) % 10 == 0):
                    print("epoch no = {}, loss={:.7f}".format(single_epoch+1, loss))
                
    
    #defining the partial fit method for fwd pass and bwd propagation
    def fit_fwd_bwd_partial(self, x, y):
        #for the first layer, the activations will be the data item itself (Eg:0,0,1)
        #conver to 2d matrix if its not 2d
        Layer_Activations = [np.atleast_2d(x)]
        
        #Feed forward Mechanism
        #loop through the rest of layers and find the activations for those layers
        for invididual_layer in np.arange(0, len(self.W)):
            #step 1: find the dot product of input feature and weight
            dot_product = Layer_Activations[invididual_layer].dot(self.W[invididual_layer])
            #step 2: pass dot product to sigmoid function
            sigmoid_out = self.sigmoid_activation_function(dot_product)
            #step 3: append the obtained value to list of activations
            Layer_Activations.append(sigmoid_out)
        
        #backpropagation mechanism
        #find the error by calculating the diff between prediction and truth value
        output_error = Layer_Activations[-1] - y
        #find the delta of the last layer and add it to the list of delta
        Delta_List = [output_error * self.deriv_sigmoid_activation_function(Layer_Activations[-1])]
        #loop through the rest of layers in reverse and find the deltas for those layers
        for invididual_layer in np.arange(len(Layer_Activations)-2, 0, -1):
            #delta for current layer = delta of prev layer . 
            #(Transpose) Weight Matrix of current layer] * 
            #[sigmoid_derivative(Current Layer activation)]
            delta_value = Delta_List[-1].dot(self.W[invididual_layer].T)
            delta_value = delta_value * self.deriv_sigmoid_activation_function(Layer_Activations[invididual_layer])
            Delta_List.append(delta_value)
        
        #reverse the list of deltas
        Delta_List = Delta_List[::-1]
        #loop through every layer and update the weight
        for invididual_layer in np.arange(0, len(self.W)):
            #update the weight for current layer
            self.W[invididual_layer] += -self.learning_rate * Layer_Activations[invididual_layer].T.dot(Delta_List[invididual_layer])
        
        
    #define the predict_evaluation method
    def predict_eval(self, X, include_bias=True):
        #convert to 2d matrix if its not 2d
        prediction = np.atleast_2d(X)
        if include_bias:
            #add a one column to the input feature matrix as bias (bias trick)
            prediction = np.c_[prediction, np.ones((prediction.shape[0]))]        
        #find the prediction and return it
        #loop through the every layer and find the prediction for those layers
        for invididual_layer in np.arange(0, len(self.W)):
            #step 1: find the dot product of input feature and weight
            prediction = prediction.dot(self.W[invididual_layer])
            #step 2: pass dot product to sigmoid function
            prediction = self.sigmoid_activation_function(prediction)   
            
        #return the prediction
        return prediction
        

    #define the find_loss method
    def find_loss(self, X, Y):
        #convert to 2d matrix if its not 2d
        actual_output = np.atleast_2d(Y)
        #find the predictions
        predicted_output = self.predict_eval(X, include_bias=False)
        #find the error
        calculated_error = predicted_output - actual_output
        #calculate loss
        loss_value = 0.5 * np.sum(calculated_error ** 2)
        #retrun loss
        return loss_value
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
