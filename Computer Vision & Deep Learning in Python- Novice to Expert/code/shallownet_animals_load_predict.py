# -*- coding: utf-8 -*-
"""
@author: abhilash
"""

from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import cv2

#load the pretrained model
model = load_model("shallownet_animals.hdf5")
print("Loaded the model from computer")
print(model.summary())

#the labels list
animals_classes = ["dog", "cat", "panda"]

#load the image to predict
img_path = 'images/test2.jpg'
img = load_img(img_path)
#resize image to square 32x32
img = img.resize((32,32))
#convert image to array
img_array = img_to_array(img)
#reshape array to 1D format
img_array = img_array.reshape((1,) + img_array.shape)

# prepare image data like as in training data
img_array = img_array.astype('float32')
img_array = img_array / 255.0

#do the prediction
predicted_output = model.predict(img_array)
#getting the maximum valued string from the labels list
predicted_text = str([animals_classes[predicted_output.argmax()]])
print("The predicted object is:")
print(predicted_text)

#display image and the prediction text over it
disp_img = cv2.imread(img_path)
#display prediction text over the image
cv2.putText(disp_img, predicted_text, (20,20), cv2.FONT_HERSHEY_TRIPLEX , 0.8, (255,0,0))
#show the image
cv2.imshow("Prediction",disp_img)
