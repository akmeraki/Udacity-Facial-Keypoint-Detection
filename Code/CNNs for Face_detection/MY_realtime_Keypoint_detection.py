import cv2
import time 
import detect_face_keypoints from FaceCNN.py

def detect_from_model(image):
    
    face_keypoints, image_with_detections = detect_face_keypoints(image)

    for face in face_keypoints:
        for x, y in zip(face[0], face[1]):
            cv2.circle(image_with_detections, (x, y), 5, (0,255,0), -1)
        
    return image_with_detections

# Capturing frames from the webcam ---> Using video_capture class from opencv 0--for webcam 1--for external camera
video_capture = cv2.VideoCapture(0)
count = 0
#Repeats infinitely
while True:
    # The we won't get the first element if we use the underscore(_)
    # The read method outputs two elements--> second one is the last frame
    _,frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect_from_model(frame)
    cv2.imshow('Video',canvas)
    #cv2.imwrite("frame%d.jpg" % count, canvas)
    #count = count + 1  # for frames 
    # if we press q the face detection stops
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
