# -*- coding: utf-8 -*-
"""
@author: abhilash
"""
from keras.preprocessing.image import img_to_array

class ImageToArray:
    def __init__(self, dataFormat=None):
        self.dataFormat = dataFormat
    
    def preprocess(self, image):
        #return the arranged image array
        return img_to_array(image, data_format=self.dataFormat)
