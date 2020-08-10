import cv2

KNOWN_WIDTH = 17.1
FOCAL_LENGTH = 672.98245614

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

while True:
    retval, frame = cap.read()
    
    #exit the application if frame not found
    if not retval:
        print("Can't receive frame (stream end?). Exiting ...")
        break 

    PIXEL_WIDTH = find_marker(frame)

    actual_distance = (KNOWN_WIDTH*FOCAL_LENGTH)/PIXEL_WIDTH
    print("actual_distance:",actual_distance)

    #cv2.putText(frame,actual_distance,(frame.shape[1] - 200, frame.shape[0] - 20),
    #                                cv2.FONT_HERSHEY_SIMPLEX,2.0, (0, 255, 0), 3)

    cv2.imshow('CalibrationWindow', frame)

        
    key = cv2.waitKey(1)
    if key == 27:
        break


#releasing the VideoCapture object
cap.release()
cv2.destroyAllWindows()
