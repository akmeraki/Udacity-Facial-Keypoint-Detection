## Canny Edge Detection

import cv2
import numpy as np
import matplotlib.pyplot as plt 

# Loading in the image
image = cv2.imread('images/fawzia.jpg')

# Converting to RGB colorspace
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Converting to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  

# Performing Canny edge detection
edges = cv2.Canny(gray,100,200)

# Dilating the image to amplify edges
edges = cv2.dilate(edges, None)

# Plotting the RGB and edge-detected image
fig = plt.figure(figsize = (15,15))
ax1 = fig.add_subplot(121)
ax1.set_xticks([])
ax1.set_yticks([])

ax1.set_title('Original Image')
ax1.imshow(image)

ax2 = fig.add_subplot(122)
ax2.set_xticks([])
ax2.set_yticks([])

ax2.set_title('Canny Edges')
ax2.imshow(edges, cmap='gray')

# Loading in the image
image = cv2.imread('images/fawzia.jpg')

## Bluring the test image Using the OpenCV's filter function

# Using an averaging kernel, and a kernel width equal to 4
kernel = np.ones((4,4),np.float32)/16
dst = cv2.filter2D(image,-1,kernel)

# Converting to RGB colorspace
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

# Converting to grayscale
gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)  

# Performing Canny edge detection
edges = cv2.Canny(gray,100,200)

# Dilating the image to amplify edges
edges = cv2.dilate(edges, None)

# Plotting the RGB and edge-detected image
fig = plt.figure(figsize = (15,15))
ax1 = fig.add_subplot(121)
ax1.set_xticks([])
ax1.set_yticks([])

ax1.set_title('Original Image')
ax1.imshow(dst)

ax2 = fig.add_subplot(122)
ax2.set_xticks([])
ax2.set_yticks([])

ax2.set_title('Canny Edges')
ax2.imshow(edges, cmap='gray')
