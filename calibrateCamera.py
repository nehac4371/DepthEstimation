import cv2
import numpy as np

def find_marker(image):
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edge = cv2.Canny(blur,35,125)
    cv2.imshow('CalibrationWindo', edge)

    cnts,_= cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts_areas = [cv2.contourArea(cnt) for cnt in cnts]
    c = cnts_areas.index(max(cnts_areas))
    x,y,w,h = cv2.boundingRect(cnts[c])
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)



#livestream from the webcam 
cap = cv2.VideoCapture(0)

#name of the display window in openCV
cv2.namedWindow('CalibrationWindow')

while True:
    #capturing frame
    retval, frame = cap.read()
    
    #exit the application if frame not found
    if not retval:
        print("Can't receive frame (stream end?). Exiting ...")
        break 

    find_marker(frame)
    
    cv2.imshow('CalibrationWindow', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

#releasing the VideoCapture object
cap.release()
cv2.destroyAllWindows()
