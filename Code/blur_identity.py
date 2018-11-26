import cv2
import matplotlib.pyplot as plt
import numpy as np

# Loading in the image
image = cv2.imread('images/gus.jpg')

# Converting the image to RGB colorspace
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Displaying the image
fig = plt.figure(figsize = (6,6))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('Original Image')
ax1.imshow(image)

## TODO: Implement face detection
## TODO: Blur the bounding box around each detected face using an averaging filter and display the result

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load in the image
image = cv2.imread('images/gus.jpg')

# Convert the image to RGB colorspace
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image
fig = plt.figure(figsize = (15,15))
ax1 = fig.add_subplot(121)
ax1.set_xticks([])
ax1.set_yticks([])
#ax1.set_title('Original Image')
#ax1.imshow(image)

## Bluring the bounding box around each detected face using an averaging filter

# Converting the RGB  image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Extracting the pre-trained face detector from an xml file
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

# Detect the faces in image
faces = face_cascade.detectMultiScale(gray,1.3, 6)

# Make a copy of the orginal image to draw face detections on
image_with_detections = np.copy(image)

# Get the bounding box for each detected face
for (x,y,w,h) in faces:
    # Add a red bounding box to the detections image
    image_rec=cv2.rectangle(image_with_detections, (x,y), (x+w,y+h), (255,0,0), 3)
    
kernel = np.ones((50,50),np.float32)/2500
dst = cv2.filter2D(image[y:y+h,x:x+h],-1,kernel)    

for j in range(h):
  for i in range(w):
        image[j+y,i+x]=dst[j,i]
ax1.imshow(image)
