#import libraries
import numpy as np
import cv2
from keras.models import load_model

#load in model for predictions
best_model = load_model('./2_batch20.h5')

#set class names
class_names = ['Anne_Hathaway','Ben','Dave_Chappelle','Elizabeth_Olsen',
               'Jessica_Alba','Mark','Zac_Efron']


#load in face cascade from OpenCV
haarcascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

#instatiate camera
capture = cv2.VideoCapture(0)

while(True):
    #capture frame by frame
    ret, frame = capture.read()

    #detect face in frames
    faces = haarcascade.detectMultiScale(frame, scaleFactor=1.09, minNeighbors=5)
    for (x, y, w, h) in faces:
        # region of interest
        roi = frame[y:y+h, x:x+w]

        ##save image of roi
        img_item = 'myimage.png'
        cv2.imwrite(img_item, roi)

        #read saved images from frame
        img = cv2.imread('./myimage.png')
        img_copy = img.copy()
        #resize saved image from frame
        face_resized_img = cv2.resize(img, (175,175), interpolation = cv2.INTER_AREA)

        #make prediction on resized images
        label = best_model.predict_classes(face_resized_img.reshape(1,175,175,3))
        label_name = class_names[label[0]]

        #output predicted class name
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = label_name
        color = (255,255,255)
        stroke = 2
        cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        #output rectangle on face
        rect_color = (255,0,0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), rect_color, stroke)

    #display frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
