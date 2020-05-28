# -*- coding: utf-8 -*-
"""
@author: abhilash
"""
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from shallownetclass import ShallowNetClass
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
#create an SGD optimizer with learning rate 0.001
stochastic_gradient_descent = SGD(0.001)
shallownet_model = ShallowNetClass.buildShallowNet(width=32, height=32, channels=3, classes=10)
#compile the model
shallownet_model.compile(optimizer=stochastic_gradient_descent, loss="categorical_crossentropy", metrics=["accuracy"])
#fit the data to the model
#fit() will return a History object which contains the details like loss and accuracy which can be used to do the plot. 
history_trained_model = shallownet_model.fit(trainX, trainY, validation_data=(testX, testY), epochs=50, batch_size=32)

#saving the model
shallownet_model.save("shallownet_cifar10.hdf5")
print("saved the model as shallownet_cifar10.hdf5")

#evaluate the model and print/plot the results
#================================================
print("evaluating the model")
model_predictions = shallownet_model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), model_predictions.argmax(axis=1), target_names=targetLabels))





