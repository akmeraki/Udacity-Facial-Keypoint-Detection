import cv2
import matplotlib.pyplot as plt
import numpy as np


# Loading in the multi-face test image again
image = cv2.imread('images/test_image_1.jpg')

# Converting the image copy to RGB colorspace
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Makeing an array copy of this image
image_with_noise = np.asarray(image)

# Creating noise - here we add noise sampled randomly from a Gaussian distribution: a common model for noise
noise_level = 40
noise = np.random.randn(image.shape[0],image.shape[1],image.shape[2])*noise_level

# Adding this noise to the array image copy
image_with_noise = image_with_noise + noise

# Converting back to uint8 format
image_with_noise = np.asarray([np.uint8(np.clip(i,0,255)) for i in image_with_noise])

# Plotting the noisy image!
fig = plt.figure(figsize = (8,8))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])

ax1.set_title('Noisy Image')
ax1.imshow(image_with_noise)



# Converting the RGB  image to grayscale
gray_noise = cv2.cvtColor(image_with_noise, cv2.COLOR_RGB2GRAY)

# Extracting the pre-trained face detector from an xml file
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

# Detecting the faces in image
faces = face_cascade.detectMultiScale(gray_noise, 4, 6)

# Printing the number of faces detected in the image
print('Number of faces detected:', len(faces))

# Making a copy of the orginal image to draw face detections
image_with_detections = np.copy(image_with_noise)

# Getting the bounding box for each detected face
for (x,y,w,h) in faces:
    # Add a red bounding box to the detections image
    cv2.rectangle(image_with_detections, (x,y), (x+w,y+h), (255,0,0), 3)
    

# Displaying the image with the detections
fig = plt.figure(figsize = (8,8))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])

ax1.set_title('Noisy Image with Face Detections')
ax1.imshow(image_with_detections)

## Using the opencv's inbuilt image denoising function

# final de-noised image (should be in RGB)
denoised_image = cv2.fastNlMeansDenoisingColored(image_with_noise,None,15,15,7,21)


# Convert the RGB  image to grayscale
gray = cv2.cvtColor(denoised_image, cv2.COLOR_RGB2GRAY)

# Extract the pre-trained face detector from an xml file
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

# Detect the faces in image
faces = face_cascade.detectMultiScale(gray, 4, 6)

# Print the number of faces detected in the image
print('Number of faces detected:', len(faces))

# Make a copy of the orginal image to draw face detections on
image_with_detections = np.copy(image_with_noise)

# Get the bounding box for each detected face
for (x,y,w,h) in faces:
    # Add a red bounding box to the detections image
    cv2.rectangle(image_with_detections, (x,y), (x+w,y+h), (255,0,0), 3)
    

# Display the image with the detections
fig = plt.figure(figsize = (8,8))
ax2= fig.add_subplot(111)
ax2.set_xticks([])
ax2.set_yticks([])

ax2.set_title('deNoised Image with Face Detections')
ax2.imshow(image_with_detections)

