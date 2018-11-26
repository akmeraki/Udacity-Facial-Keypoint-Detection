import cv2
import time 
from keras.models import load_model
import numpy as np
import sunglasses_over_image from Sunglasses_filter.py
# Load facial landmark detector model
model = load_model('my_model.h5')

# Capturing frames from the webcam ---> Using video_capture class from opencv 0--for webcam 1--for external camera
video_capture = cv2.VideoCapture(0)
count = 0
#Repeats infinitely
while True:
    # The we won't get the first element if we use the underscore(_)
    # The read method outputs two elements--> second one is the last frame
    _,frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output = sunglasses_over_image(frame)
    cv2.imshow('Video',output)
    cv2.imwrite("frame%d.jpg" % count, output)
    count = count + 1
    # if we press q the face detection stops
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
