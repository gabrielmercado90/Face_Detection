import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np

#clasificacion de cara
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier()
f = 0

video = cv2.VideoCapture(0)
while video.isOpened():
    #captura del video
    ret, frame = video.read()
    roi = None
    if frame is not None:
        #deteccion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            #region de interes
            roi = frame[y + 2:y + h - 2, x + 2: x + w - 2]
            plt.imshow(roi)
            cv2.rectangle(frame, (x,y), (x + w, y + h), (255,0,0), 2)
            
        #mostrar
        cv2.imshow('capture', frame) 
        f = f+1
        
        print(f) #numero de frames

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and roi is not None: cv2.imwrite('face_capture.jpg', roi)
    elif key == ord('q'): break


video.release()
cv2.destroyAllWindows()
