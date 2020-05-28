import cv2

image = cv2.imread('images/cat.png')
print(image.shape)
cv2.imshow("Test Image", image)
print(image[50,30])