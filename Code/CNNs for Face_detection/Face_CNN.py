from utils import load_data,plot_data,plot_keypoints
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import keras 
import tensorflow

# Loading training set
X_train, y_train = load_data(test=False)
print("X_train.shape == {}".format(X_train.shape))
print("y_train.shape == {}; y_train.min == {:.3f}; y_train.max == {:.3f}".format(y_train.shape, y_train.min(), y_train.max()))

# Loading testing set
X_test, _ = load_data(test=True)
print("X_test.shape == {}".format(X_test.shape))

## Visualizing the Trainging Data 
## For each training image, there are two landmarks per eyebrow (four total), three per eye (six total), four for the mouth, and one for the tip of the nose.


fig = plt.figure(figsize=(20,20))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(9):
    ax = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])
    plot_data(X_train[i], y_train[i], ax) 


### CNN Architecture

# Importing deep learning resources from Keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense


##CNN architecture
# Model should accept 96x96 pixel graysale images in
# It should have a fully-connected output layer with 30 values (2 for each facial keypoint)

model = Sequential()
model.add(Convolution2D(filters=32,kernel_size=3,input_shape=(96,96,1)))
model.add(Convolution2D(filters=64,kernel_size=3,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100,activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(30,activation='tanh'))

# Summarizing the model
model.summary()

##Compile and Train the Model
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.callbacks import ModelCheckpoint


model.compile(loss='mean_squared_error',optimizer='adam')

## Training the model
# Saving the model as a HDF5 file 
checkpointer= ModelCheckpoint(filepath='my_model.h5', verbose=2, save_best_only=True)

history = model.fit(X_train, y_train, batch_size=64, epochs=40, validation_split=0.2,callbacks=[checkpointer], verbose=2,shuffle=True)


## Visualizing the Loss and test

# Visualizing the training and validation loss of the neural network
# summarizing history for loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#Visualizing a Subset of the Test Predictions
y_test = model.predict(X_test)
fig = plt.figure(figsize=(20,20))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(9):
    ax = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])
    plot_data(X_test[i], y_test[i], ax)

## Completing the Computer Vision Pipeline

# Loading the color image for face detection
image = cv2.imread('images/obamas4.jpg')

# Converting the image to RGB colorspace
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_copy=np.copy(image)

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# plotting the image
fig = plt.figure(figsize = (9,9))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('image copy')
ax1.imshow(image_copy)

### Using the face detection code with the trained conv-net 
## Painting the predicted keypoints on the test image


def detect_face_keypoints(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 2, 6)
    image_with_detections = np.copy(image)

    face_keypoints = []
    for (x,y,w,h) in faces:
        face_img = image_with_detections[y:y+h, x:x+w]

        face_cropped= cv2.resize(face_img, (96, 96))
        gray = cv2.cvtColor(face_cropped, cv2.COLOR_RGB2GRAY)

        gray_normal = gray / 255.
        gray_normal = gray_normal[np.newaxis, :, :, np.newaxis]

        key_points = model.predict(gray_normal)
        key_points = key_points * 48 + 48
        
    
        x_coord = key_points[0][0::2]
        y_coord = key_points[0][1::2]
        x_coord = x_coord * w / 96 + x
        y_coord = y_coord * h / 96 + y

        cv2.rectangle(image_with_detections, (x,y), (x+w,y+h), (255,0,0), 3)

        face_keypoints.append((x_coord, y_coord))
        
    return face_keypoints, image_with_detections


# Plotting the Detected face keypoints in image
face_keypoints, image_with_detections = detect_face_keypoints(image)

for face in face_keypoints:
    for x, y in zip(face[0], face[1]):
        cv2.circle(image_with_detections, (x, y), 5, (0,255,0), -1)

plt.imshow(image_with_detections, cmap='gray')



