import cv2
import numpy as np
import matplotlib.pyplot as plt

# Call the laptop camera face/eye detector function below
# Loading the Haar cascades Classifier
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_eye.xml')

def detect(gray,frame):
    """
	Detects the face and eyes in a frame
    Argument:
	gray -- a grayscale image of the frame
	frame -- the original frame from the webcam
	Returns:
	frame_face -- The detected image of the face
    frame_eyes -- The detected image of the eyes
	"""
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        # Drawing the rectangle --> frame, top-left co-ordinate, bottom-right co-ordinate, Color, Thickness
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        # Detecting the eyes ---> using roi_gray just to reduce computation
        eyes = eyes_cascade.detectMultiScale(roi_gray, 1.1 ,3)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return frame

# Capturing frames from the webcam ---> Using video_capture class from opencv 0--for webcam 1--for external camera
video_capture = cv2.VideoCapture(0)

#Repeats infinitely
while True:
    # we won't get the first element if we use the underscore(_)
    # The read method outputs two elements--> second one is the last frame
    _,frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray,frame)
    cv2.imshow('Video',canvas)
    # if we press q the face detection stops
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
