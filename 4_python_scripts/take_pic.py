import numpy as np
import cv2

#load face cascade
haarcascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

capture = cv2.VideoCapture(0)


i = 1
while(True):
    #capture frame by frame
    ret, frame = capture.read()

    faces = haarcascade.detectMultiScale(frame, scaleFactor=1.09, minNeighbors=5)
    for (x, y, w, h) in faces:
        # print (x,y,w,h)
        roi = frame[y:y+h, x:x+w]

        #resize images to model image size
        resized_roi = cv2.resize(frame[y:y+h, x:x+w], (175,175), interpolation = cv2.INTER_AREA)

        #save images
        img_item = "%i.jpg"%i
        cv2.imwrite(img_item, resized_roi)
        i += 1


    #display frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
