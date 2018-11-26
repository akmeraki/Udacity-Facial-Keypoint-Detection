import numpy as np
import matplotlib.pyplot as plt
import math
import cv2                  # OpenCV library for computer vision
from PIL import Image
import time 

# Loading the color image for face detection
image = cv2.imread('images/james.jpg')

# Converting the image to RGB colorspace
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Plotting the RGB image
fig = plt.figure(figsize = (6,6))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])

ax1.set_title('Original Image')
ax1.imshow(image)

# Converting the RGB  image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Extracting the pre-trained face detector from an xml file
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

# Detecting the faces in image
faces = face_cascade.detectMultiScale(gray, 1.25, 6)

# Printing the number of faces detected in the image
print('Number of faces detected:', len(faces))

# Making a copy of the orginal image to draw face detections on
image_with_detections = np.copy(image)


# Displaying the bounding box for each detected face
for (x,y,w,h) in faces:
    # Add a red bounding box to the detections image
    cv2.rectangle(image_with_detections, (x,y), (x+w,y+h), (255,0,0), 3)
    
# Displaying the image with the detections
fig = plt.figure(figsize = (6,6))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])

ax1.set_title('Image with Face Detection')
ax1.imshow(image_with_detections)


## Adding Eye Detection

# Making a copy of the original image to plot rectangle detections
image_with_detections = np.copy(image) 

eye_cascade=cv2.CascadeClassifier('detector_architectures/haarcascade_eye.xml')

# Looping over the detections and drawing their corresponding face detection boxes
for (x,y,w,h) in faces:
    cv2.rectangle(image_with_detections, (x,y), (x+w,y+h),(255,0,0), 3)  
    

    eyes=eye_cascade.detectMultiScale(gray)
    for (a,b,c,d) in eyes:
        image_with_detections=cv2.rectangle(image_with_detections,(a,b),(a+c,b+d),(0,255,0),3)
    

# Plotting the image with both faces and eyes detected
fig = plt.figure(figsize = (6,6))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])

ax1.set_title('Image with Face and Eye Detection')
ax1.imshow(image_with_detections)
