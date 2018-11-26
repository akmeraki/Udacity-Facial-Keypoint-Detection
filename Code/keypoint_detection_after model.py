import cv2
import numpy as np
import matplotlib.pyplot as plt

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


face_keypoints, image_with_detections = detect_face_keypoints(image)
for face in face_keypoints:
    for x, y in zip(face[0], face[1]):
        cv2.circle(image_with_detections, (x, y), 5, (0,255,0), -1)

plt.imshow(image_with_detections, cmap='gray')
