# -*- coding: utf-8 -*-
"""
@author: abhilash
"""
from backpropagation import BackPropagation
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets


#loading the subset/sample MNIST dataset
print("Loading the MNIST sample dataset")
numbers = datasets.load_digits()
#covert the data in the dataset to datatype float
digits_data = numbers.data.astype("float")
#labels for the data
digits_labels = numbers.target

#normalize so that the data will be within 0 to 1 range
digits_data = (digits_data - digits_data.min()) / (digits_data.max() - digits_data.min())
#print the shape of dataset
print("No of samples: {}, data set dimension: {}".format(digits_data.shape[0],digits_data.shape[1]))


#splitting the dataset 75% for training and 25% for testing
(trainX, testX, trainY, testY) = train_test_split(digits_data, digits_labels, test_size=0.25)

#binarize the labels
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

#create and train the network
print("Creating and training the network ")
MnistNeuralNetwork = BackPropagation([trainX.shape[1], 32, 16, 10])
MnistNeuralNetwork.train_fit(trainX, trainY, no_of_epochs=1000)

#test and evaluate the network
print("Testing and evaluating the network")
label_predictions = MnistNeuralNetwork.predict_eval(testX)
#get the maximum valued label as the prediction
label_predictions = label_predictions.argmax(axis=1)
#print the classification report
#testing output vs predicted output
print(classification_report(testY.argmax(axis=1), label_predictions))









