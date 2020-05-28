# -*- coding: utf-8 -*-
"""
@author: abhilash
"""
from sklearn.preprocessing import LabelBinarizer
from vggnetclass import VGGNetClass
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint



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

#declare the filename template string for keras
file_name = "vggnet_cifar10-{epoch:03d}-{val_accuracy:.4f}.hdf5"
#instantiate ModelCheckpoint class
improvement_checkpoint = ModelCheckpoint(file_name, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1)
#define call backs list
improvement_callback = [improvement_checkpoint]

#include callback for improvment checkpoint
#fit the data to the model
#fit() will return a History object which contains the details like loss and accuracy which can be used to do the plot. 
history_trained_model = vggnet_model.fit(trainX, trainY, validation_data=(testX, testY), epochs=3, batch_size=64, callbacks=improvement_callback)


