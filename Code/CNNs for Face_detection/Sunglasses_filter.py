import cv2
import matplotlib.pyplot as plt
import numpy as np
 
# image has a 4th channel that allows us to control how transparent each pixel in the image is
sunglasses = cv2.imread("images/sunglasses_4.png", cv2.IMREAD_UNCHANGED)

# Plotting the image
fig = plt.figure(figsize = (6,6))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.imshow(sunglasses)
ax1.axis('off')

# Printing out the shape of the sunglasses image
print ('The sunglasses image has shape: ' + str(np.shape(sunglasses)))

# Printing out the sunglasses transparency (alpha) channel
alpha_channel = sunglasses[:,:,3]
print ('the alpha channel here looks like')
print (alpha_channel)

# Just to double check that there are indeed non-zero values
values = np.where(alpha_channel != 0)
print ('\n the non-zero values of the alpha channel look like')
print (values)

# Loading in color image for face detection
image = cv2.imread('images/obamas4.jpg')

# Converting the image to RGB colorspace
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Plotting the image
fig = plt.figure(figsize = (6,6))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('Original Image')
ax1.imshow(image)

#Reshaping the Sunglasses before warping 
reshaped_sunglasses = cv2.resize(sunglasses, (image.shape[1], image.shape[0]))
print(reshaped_sunglasses.shape)
plt.imshow(reshaped_sunglasses[:,:,:3])


# Warping the Glasses using Warp function

def warp(image, dest_points):

    src_points = np.float32([[0, 0],
                             [0, 499],
                             [758, 499],
                             [758, 0]])
    
    M = cv2.getPerspectiveTransform(src_points, dest_points)
    image_size = (image.shape[1], image.shape[0])
    warped_image = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)
    return warped_image


## Using the face detection code with the trained conv-net to put sunglasses on the individuals in the test image

def sunglasses_over_image(image):
    face_keypoints, image_with_detections = detect_face_keypoints(image)
    image_with_sunglasses = np.copy(image)
    output = np.copy(image)

    for points in face_keypoints:
        points_x, points_y = points
        h = (points_y[5] - points_y[9]) * 3

        destination_pts = np.float32([[points_x[9], points_y[9]],
                               [points_x[9], points_y[9]+h],
                               [points_x[7], points_y[9]+h],
                               [points_x[7], points_y[9]]])

        warped_sunglasses = warp(reshaped_sunglasses, destination_pts)

        mask = warped_sunglasses[:,:,:3]
        mask[mask == 0] = 255

        output[mask != 255] = mask[mask != 255]

    return output 

#Plotting the output
output = sunglasses_over_image(image)
plt.imshow(output)
