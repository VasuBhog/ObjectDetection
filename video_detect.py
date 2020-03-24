import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from cvlib.object_detection import draw_bbox

#Video File Capture
cap = cv.VideoCapture('/Users/vasubhog/ObjectDetection/assets/1_car.mp4')

# while cap.isOpened():

#     #Width x Height
#     cap.get(cv.CAP_PROP_FRAME_WIDTH)
#     cap.get(cv.CAP_PROP_FRAME_HEIGHT)

#     ret, frame = cap.read()
#     # if frame is read correctly ret is True
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     cv.imshow('frame', gray)
#     if cv.waitKey(1) == ord('q'):
#         break
# cap.release()
# cv.destroyAllWindows()