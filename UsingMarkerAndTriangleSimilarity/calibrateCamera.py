import cv2
import numpy as np
from scipy import stats

KNOWN_DISTANCE = 84.0
KNOWN_WIDTH = 17.1

def find_marker(image):
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edge = cv2.Canny(blur,35,90)
    
    cnts,_= cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts_areas = [cv2.contourArea(cnt) for cnt in cnts]
    c = cnts_areas.index(max(cnts_areas))
    x,y,w,h = cv2.boundingRect(cnts[c])
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

    return w


#livestream from the webcam 
cap = cv2.VideoCapture(0)

#name of the display window in openCV
cv2.namedWindow('CalibrationWindow')

#Focal Length 
focalLength = []

i = 0

while i<1000:
    #capturing frame
    retval, frame = cap.read()
    
    #exit the application if frame not found
    if not retval:
        print("Can't receive frame (stream end?). Exiting ...")
        break 

    PIXEL_WIDTH = find_marker(frame)

    focalLength.append((PIXEL_WIDTH*KNOWN_DISTANCE)/KNOWN_WIDTH)
    i+=1
    
    cv2.imshow('CalibrationWindow', frame)

#releasing the VideoCapture object
cap.release()
cv2.destroyAllWindows()


print("Number of Readings:",len(focalLength))
#average_focal_length = sum(focalLength)/len(focalLength)
mode_focal_length = stats.mode(np.array(focalLength))
#print("average Focal Length:",average_focal_length)
print("mode Focal Length:",mode_focal_length)


