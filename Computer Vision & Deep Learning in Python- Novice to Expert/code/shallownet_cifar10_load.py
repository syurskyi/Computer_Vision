# -*- coding: utf-8 -*-
"""
@author: abhilash
"""
from keras.models import load_model

model = load_model("shallownet_cifar10.hdf5")
print("Loaded the model from computer")
print(model.summary())

