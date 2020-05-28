# -*- coding: utf-8 -*-
"""
@author: abhilash
"""

from keras.models import load_model
from PIL import Image
import numpy as np

#load the pretrained model
model = load_model("lenet_minist.hdf5")
print("Loaded the model from computer")
print(model.summary())


#load the image to predict
img_path = 'images/handwritten_numbers/nine.png'
img = Image.open(img_path).convert("L")
#resize image to square 28x28
img = img.resize((28,28))
#convert image to array
img_array = np.array(img)
#reshape array to 1D format
img_array = img_array.reshape(1, 28, 28, 1)

# prepare image data like as in training data
img_array = img_array.astype('float32')
img_array = img_array / 255.0

#do the prediction
predicted_output = model.predict(img_array)
#getting the maximum valued string from the labels list
predicted_text = predicted_output.argmax()
print("The predicted digit is:")
print(predicted_text)
