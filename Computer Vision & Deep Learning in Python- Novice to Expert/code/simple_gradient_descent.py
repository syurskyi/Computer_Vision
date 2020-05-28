# -*- coding: utf-8 -*-
"""

@author: abhilash
"""
#import the required packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#define learning rate and epochs
learning_rate = 0.001
no_of_epochs = 100

#generate a sample data set with two feature columns and one class column
# features with random numbers and class with either 0 or 1
# total number of samples will be 1000
(X,y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)

#convert shape of y into a double indexed form (1000,) => (1000,1)
y = y.reshape((y.shape[0], 1))

#reshape X value to include a 1's column to accomodate the weight column due to 
# the bias trick
X = np.c_[X, np.ones((X.shape[0]))]

#split 50% (500 rows) for training and rest 50% for testing
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=50)

# initialize a 3x1 matrix with random values as Weight Matrix 
W = np.random.randn(X.shape[1], 1)


# initialize a list for storing loss values during epochs
losses_value = []

#define sigmoid activation function
def sigmoid_activation_function(x):
    #calculating the sigmoid activation value
    return 1.0/(1+np.exp(-x))

#define predict function
def predict_function(X, W):
    prediction = sigmoid_activation_function(X.dot(W))
    #use step function to convert the prediction to class labels 1 or 0
    prediction[prediction <= 0.5] = 0
    prediction[prediction > 0.5] = 1
    return prediction

#starting the training epochs
print("Starting training epochs")
#looping the number of epochs
for epoch in np.arange(0,no_of_epochs):
    predictions = sigmoid_activation_function(trainX.dot(W))
    # to find error , substract the true output from the predicted output
    error = predictions - trainY
    #find the loss value and append it to the losses_value list
    loss_value = np.sum(error ** 2)
    losses_value.append(loss_value)
    #find the slope (gradient), dot product of training input (trasposed) and current error
    gradient = trainX.T.dot(error)
    #add to the existing value of Weight W, the new variation
    #using the negative gradient (the descending gradient)
    W += -(learning_rate) * gradient
    print("Epoch Number:{}, loss:{:.7f}".format(int(epoch+1),loss_value))


#starting the testing/evaluation 
print("Starting testing/evaluation")
#obtain predictions by using testing input data and the 
#already computed and updated Weight value W
predictions =  predict_function(testX, W)
#give report by comparing predictions and the truth value testY
print(classification_report(testY, predictions))


#plotting the data set as scatter plot
plt.style.use("ggplot")
plt.figure()
plt.title("Scatter plot of dataset")
plt.scatter(testX[:,0], testX[:,1])

#plotting the error vs epochs graph
plt.style.use("ggplot")
plt.figure()
plt.title("Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(np.arange(0,no_of_epochs), losses_value)







