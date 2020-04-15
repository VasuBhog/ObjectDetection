import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from cvlib.object_detection import draw_bbox

#load image
image = cv.imread('assets/road_1.jpg')
cv.imshow('image',image)
cv.waitKey(0)
cv.destroyAllWindows()

#Gray scale image
imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("grey",imgray)
cv.waitKey(0)
cv.destroyAllWindows()

#hsv 
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
cv.imshow("grey",imgray)
cv.waitKey(0)
cv.destroyAllWindows()

#threshold - yellow
lower_yellow = np.array([20,100,100])
upper_yellow = np.array([30,255,255])

#mask for yellow license plate
mask = cv.inRange(hsv,lower_yellow,upper_yellow)
cv.imshow("mask",mask)
cv.waitKey(0)
cv.destroyAllWindows()

#contour
contours = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
print(contours)
cv.drawContours(image,contours, -1,(0,255,0), 3)