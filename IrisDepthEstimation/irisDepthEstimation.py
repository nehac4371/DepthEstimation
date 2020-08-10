import cv2
import numpy as np
import dlib

#The average diameter of the human iris is 1.17+-0.05 cm
KNOWN_WIDTH = 1.17

#The focal length calculated from calibration
FOCAL_LENGTH = 522.72727273


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("IrisDepthEstimation\shape_predictor_68_face_landmarks.dat")
font = cv2.FONT_HERSHEY_SIMPLEX

left_eye_landmarks  = [36, 37, 38, 39, 40, 41]
right_eye_landmarks = [42, 43, 44, 45, 46, 47]



def get_eye_points(facial_landmarks,eye_landmarks):

    left_top = (facial_landmarks.part(eye_landmarks[1]).x, 
                facial_landmarks.part(eye_landmarks[1]).y)
    
    right_bottom = (facial_landmarks.part(eye_landmarks[4]).x, 
                    facial_landmarks.part(eye_landmarks[4]).y)

    return (left_top,right_bottom)      


#livestream from the webcam 
cap = cv2.VideoCapture(0)

#name of the display window in openCV
cv2.namedWindow('CalibrationWindow')

while True:
    retval, frame = cap.read()
    
    #exit the application if frame not found
    if not retval:
        print("Can't receive frame (stream end?). Exiting ...")
        break 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces,_,_ = detector.run(gray,0,0)
    facial_landmarks = predictor(gray, faces[0])
    
    left_eye_points = get_eye_points(facial_landmarks,left_eye_landmarks)
    cv2.rectangle(frame,left_eye_points[0],left_eye_points[1],(255,0,0),1)

    right_eye_points = get_eye_points(facial_landmarks,right_eye_landmarks)
    cv2.rectangle(frame,right_eye_points[0],right_eye_points[1],(255,0,0),1)

    left_iris_diameter =  (abs(left_eye_points[0][0]-left_eye_points[1][0])
                          +abs(left_eye_points[0][1]-left_eye_points[1][1]))/2

    right_iris_diameter = (abs(right_eye_points[0][0]-right_eye_points[1][0])
                          +abs(right_eye_points[0][1]-right_eye_points[1][1]))/2


    #print('left_iris_diameter:',left_iris_diameter)
    #print('right_iris_diameter:',right_iris_diameter)
    PIXEL_WIDTH = (left_iris_diameter+right_iris_diameter)/2
    
    actual_distance = (KNOWN_WIDTH*FOCAL_LENGTH)/PIXEL_WIDTH
    print("actual_distance:",actual_distance)

    cv2.imshow('CalibrationWindow', frame)

        
    key = cv2.waitKey(1)
    if key == 27:
        break


#releasing the VideoCapture object
cap.release()
cv2.destroyAllWindows()