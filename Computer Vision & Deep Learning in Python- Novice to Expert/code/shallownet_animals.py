# -*- coding: utf-8 -*-
"""
@author: abhilash
"""
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from shallownetclass import ShallowNetClass
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imutils import paths
from dsloader import DsLoader
from dspreprocessor import DsPreprocessor
from imagetoarray import ImageToArray
import numpy as np


# get the list of images from the dataset path
image_paths = list(paths.list_images('datasets/animals'))

print("INFO: loading and preprocessing")
#loading and preprocessing images using the classes created
# create instances for the loader and preprocessor classes
dp = DsPreprocessor(32, 32)
itoa = ImageToArray()
dl = DsLoader(preprocessors=[dp, itoa])
(data, labels) = dl.load(image_paths)
#normalizing the array of data
data = data.astype("float")/255.0

print("INFO: splitting the dataset")
# split 25 percentage for testing and rest for training
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=40)
#binarization using one hot encoding
label_binarizer = LabelBinarizer()
trainY = label_binarizer.fit_transform(trainY)
testY = label_binarizer.fit_transform(testY)

#train the model using ShallowNet and Stochastic Gradient Descent optimizer
#================================================
print("training the model")
#create an SGD optimizer with learning rate 0.005
stochastic_gradient_descent = SGD(0.005)
shallownet_model = ShallowNetClass.buildShallowNet(width=32, height=32, channels=3, classes=3)
#compile the model
shallownet_model.compile(optimizer=stochastic_gradient_descent, loss="categorical_crossentropy", metrics=["accuracy"])
#fit the data to the model
#fit() will return a History object which contains the details like loss and accuracy which can be used to do the plot. 
history_trained_model = shallownet_model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=32)


#evaluate the model and print/plot the results
#================================================
print("evaluating the model")
model_predictions = shallownet_model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), model_predictions.argmax(axis=1), target_names=["cat", "dog", "panda"]))


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


