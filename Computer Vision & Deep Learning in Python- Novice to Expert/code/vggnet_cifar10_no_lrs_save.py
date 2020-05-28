# -*- coding: utf-8 -*-
"""
@author: abhilash
"""
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from vggnetclass import VGGNetClass
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np


print("INFO: loading the cifar-10 dataset")
((trainX, trainY), (testX, testY)) = cifar10.load_data()

#normalize the pixel intesity of training data to fall within 0 to 1
trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0

#binarization using one hot encoding
label_binarizer = LabelBinarizer()
trainY = label_binarizer.fit_transform(trainY)
testY = label_binarizer.transform(testY)

#the target labels list
targetLabels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

#train the model using ShallowNet and Stochastic Gradient Descent optimizer
#================================================
print("training the model")
#create an SGD optimizer with learning rate 0.01
stochastic_gradient_descent = SGD(lr=0.01)
vggnet_model = VGGNetClass.buildVGGNet(width=32, height=32, channels=3, classes=10)
#compile the model
vggnet_model.compile(optimizer=stochastic_gradient_descent, loss="categorical_crossentropy", metrics=["accuracy"])
#fit the data to the model
#fit() will return a History object which contains the details like loss and accuracy which can be used to do the plot. 
history_trained_model = vggnet_model.fit(trainX, trainY, validation_data=(testX, testY), epochs=40, batch_size=64)

#saving the model
vggnet_model.save("vggnet_cifar10_no_lrs.hdf5")
print("saved the model as vggnet_cifar10_no_lrs.hdf5")

#evaluate the model and print/plot the results
#================================================
print("evaluating the model")
model_history = vggnet_model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), model_history.argmax(axis=1), target_names=targetLabels))


#plot the graph of training vs accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,40), model_history.history["loss"], label="train_loss")
plt.plot(np.arange(0,40), model_history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,40), model_history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0,40), model_history.history["val_accuracy"], label="val_acc")
plt.title("No LRS - Loss/Accuracy vs Epochs")
plt.xlabel("No of Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend()



