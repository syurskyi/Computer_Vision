# -*- coding: utf-8 -*-
"""

@author: abhilash
"""

import numpy as np
import cv2
from skimage.exposure import rescale_intensity


def custom_convolution(input_image, kernel):
    #get the width and height of input image and kernel
    (inputHeight, inputWidth) = input_image.shape[:2]
    (kernelHeight, kernelWidth) = kernel.shape[:2]
    #create the padding and apply that to the input image as an extra border
    #formula to get the padding width = width of kernal-1 / 2 (with no reminders)
    padding = (kernelWidth-1) // 2
    padded_input_image = cv2.copyMakeBorder(input_image, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
    #create an empty place holder for the output image 
    output_image = np.zeros((inputHeight, inputWidth), dtype="float")
    
    #moving the kernel over the height and width input image
    for y_pos in np.arange(padding, inputHeight+padding):
        for x_pos in np.arange(padding, inputWidth+padding):
            #extracting the region of interest
            region_of_interest = padded_input_image[y_pos-padding:y_pos+padding+1, x_pos-padding:x_pos+padding+1]
            #calculate the convolution
            convolution = (region_of_interest*kernel).sum()
            #place that convoluted value in the respective matrix position
            output_image[y_pos-padding, x_pos-padding] = convolution
    
    #make sure to place the pixel in range 0 to 255
    output_image = rescale_intensity(output_image, in_range=(0, 255))
    #convert the data type of the result image to 8bit unsigned integer
    output_image = (output_image * 255).astype("uint8")
    #return the covoluted image
    return output_image

#define the list of filters
blur_filter = np.ones((21,21), dtype="float")*(1.0 / (21 * 21))
sharpen_filter = np.array((
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]), dtype="int")
emboss_filter = np.array((
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]), dtype="int")
sobel_filter = np.array((
        [1, 0, 1],
        [2, 0, 2],
        [1, 0, 1]), dtype="int")

#loading the image
my_image = cv2.imread('C:\\ABHIS\\tech\\Deep Learning using CV\\code\\images\\cat_greyscale.jpg')
#convert color image to greyscale using opencv
gray_my_image = cv2.cvtColor(my_image, cv2.COLOR_BGR2GRAY)
#apply filter to the image
filtered_image = custom_convolution(gray_my_image, sobel_filter)
#display the filtered image
cv2.imshow("Filtered Image", filtered_image)
    
    
    