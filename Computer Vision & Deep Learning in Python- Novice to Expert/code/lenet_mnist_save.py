# -*- coding: utf-8 -*-
"""

@author: abhilash
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from lenetclass import LeNetClass

#get the MNIST full dataset
print("fetching the MNIST dataset")
# depricated method
#mnist_dataset = datasets.fetch_mldata("MNIST Original")
mnist_dataset = fetch_openml('mnist_784', version=1, cache=True)
#fetch_openml will return targets as string, convert to 8 bit int
mnist_dataset.target = mnist_dataset.target.astype(np.int8)

#since this is a 'channels last' image data format, reshape the dataset to
# (no of samples, width, height, channels)
mnist_data = mnist_dataset.data.reshape(mnist_dataset.data.shape[0], 28, 28, 1)

#normalize the data from 0-255 range to 0 to 1 range
mnist_data = mnist_data.astype("float") / 255.0

#split the dataset, 75% for training and 25% for testing
(trainX, testX, trainY, testY) = train_test_split(mnist_data, mnist_dataset.target, test_size=0.25)

#binarization using one hot encoding
label_binarizer = LabelBinarizer()
trainY = label_binarizer.fit_transform(trainY)
testY = label_binarizer.transform(testY)

#define the model using lenet class
print("defining the model")
mnist_model = LeNetClass.buildLeNet(width=28, height=28, channels=1, classes=10)

#train the model using Stochastic Gradient Descent optimizer
#================================================
print("training the model")
#create an SGD optimizer with learning rate 0.01
stochastic_gradient_descent = SGD(0.01)
#compile the model
mnist_model.compile(optimizer=stochastic_gradient_descent, loss="categorical_crossentropy", metrics=["accuracy"])
#fit the data to the model
#fit() will return a History object which contains the details like loss and accuracy which can be used to do the plot. 
model_history = mnist_model.fit(trainX, trainY, validation_data=(testX, testY), epochs=20, batch_size=128)

#save the model
mnist_model.save("lenet_mnist.hdf5")
print("Saved the model as lenet_mnist.hdf5")


#evaluate the model and print/plot the results
#================================================
print("evaluating the model")
model_predictions = mnist_model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), model_predictions.argmax(axis=1), target_names=[str(label) for label in label_binarizer.classes_]))


#plot the graph of training vs accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,20), model_history.history["loss"], label="train_loss")
plt.plot(np.arange(0,20), model_history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,20), model_history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0,20), model_history.history["val_accuracy"], label="val_acc")
plt.title("Loss/Accuracy vs Epochs")
plt.xlabel("No of Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend()
