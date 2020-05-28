# -*- coding: utf-8 -*-
"""

@author: abhilash
"""
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

#get the CIFAR-10  dataset
print("fetching the CIFAR-10 dataset")

#load the pre splitted cifar10 dataset
((trainX, trainY), (testX, testY)) = cifar10.load_data()

#normalize the train and test input data from 0-255 range to 0 to 1 range
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

print("shape of trainX ", trainX.shape)
print("shape of testX ", testX.shape)
#reshape the training and testing inputs to be passed into the network
trainX = trainX.reshape((trainX.shape[0], 3072))
testX = testX.reshape((testX.shape[0], 3072))

#binarization using one hot encoding
label_binarizer = LabelBinarizer()
trainY = label_binarizer.fit_transform(trainY)
testY = label_binarizer.transform(testY)

#define the class names as a list 
classNames = ["airplanes", "automobiles", "birds", "cats", "deer", "dogs", "frogs", "horses", "ships", "trucks"]

#define the model of shape 3072 -> 1024 -> 512 -> 10 using keras library
print("defining the model")
cifar10_model = Sequential()
cifar10_model.add(Dense(1024, input_shape=(3072,), activation="relu"))
cifar10_model.add(Dense(512, activation="relu"))
cifar10_model.add(Dense(10, activation="softmax"))

#train the model using Stochastic Gradient Descent optimizer
#================================================
print("training the model")
#create an SGD optimizer with learning rate 0.01
stochastic_gradient_descent = SGD(0.01)
#compile the model
cifar10_model.compile(optimizer=stochastic_gradient_descent, loss="categorical_crossentropy", metrics=["accuracy"])
#fit the data to the model
#fit() will return a History object which contains the details like loss and accuracy which can be used to do the plot. 
history_trained_model = cifar10_model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=32)

#evaluate the model and print/plot the results
#================================================
print("evaluating the model")
model_predictions = cifar10_model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), model_predictions.argmax(axis=1), target_names=classNames))


#plot the graph of training vs accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,100), history_trained_model.history["loss"], label="train_loss")
plt.plot(np.arange(0,100), history_trained_model.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,100), history_trained_model.history["accuracy"], label="train_acc")
plt.plot(np.arange(0,100), history_trained_model.history["val_accuracy"], label="val_acc")
plt.title("Loss/Accuracy vs Epochs")
plt.xlabel("No of Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend()
