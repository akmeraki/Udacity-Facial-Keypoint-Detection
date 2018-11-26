import cv2
import numpy as np
import matplotlib.pyplot as plt

# Loading the Haar cascades Classifier
face_cascade = cv2.CascadeClassifier('/home/akg/Desktop/Udemy/Module_1_Face_Recognition/haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('/home/akg/Desktop/Udemy/Module_1_Face_Recognition/haarcascade_eye.xml')

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
        roi_color = frame[y:y+h,x:x+w]
        kernel = np.ones((50,50),np.float32)/2500
        dst = cv2.filter2D(roi_color,-1,kernel)
        for j in range(h):
            for i in range(w):
                frame[j+y,i+x]=dst[j,i]
             
    return frame

# Capturing frames from the webcam ---> Using video_capture class from opencv 0--for webcam 1--for external camera
video_capture = cv2.VideoCapture(0)
count = 0
#Repeats infinitely
while True:
    # The we won't get the first element if we use the underscore(_)
    # The read method outputs two elements--> second one is the last frame
    _,frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray,frame)
    cv2.imshow('Video',canvas)
    cv2.imwrite("frame%d.jpg" % count, canvas)
    count = count + 1
    # if we press q the face detection stops
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
