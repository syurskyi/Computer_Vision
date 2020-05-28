# -*- coding: utf-8 -*-
"""
@author: abhilash
"""
from sklearn.preprocessing import LabelBinarizer
from vggnetclass import VGGNetClass
from keras.optimizers import SGD
from keras.datasets import cifar10



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

#print the summary of the model
print(vggnet_model.summary())