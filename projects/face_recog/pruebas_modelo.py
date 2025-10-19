import cv2 as cv
import numpy as np
import os 


absPath = '/Users/carloskvpchc/Documents/tec_projects/ai_assignature/projects/face_recog/trained_models/FisherFace2.xml'
haarcascadePath = '/Users/carloskvpchc/Documents/tec_projects/ai_assignature/haarcascade_frontalface_alt.xml'
# Etiquetas definidas para las clases al momento del entrenamiento
faces = ['charly', 'dalia', 'octavio', 'paulina']

vCap = cv.VideoCapture(0)
faceClasifier = cv.CascadeClassifier(haarcascadePath)


faceRecognizer = cv.face.FisherFaceRecognizer_create()  # type: ignore
faceRecognizer.read(absPath)

while True: 
    ret, frame = vCap.read()
    if ret == False : break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    copyGr = gray.copy()

    facesDetected = faceClasifier.detectMultiScale(gray, 1.3, 3)
    for (x, y , w, h)in facesDetected:
        
        grayFrame = copyGr[y: y + h, x: x + w]
        grayFrame = cv.resize(grayFrame, (48, 48), interpolation=cv.INTER_CUBIC)
        
        result = faceRecognizer.predict(grayFrame)
        cv.putText(frame, '{}'.format(result), (x,y-20), 1,3.3, (255,255,0), 1, cv.LINE_AA)
        if result[1] < 500:
            cv.putText(frame,'{}'.format(faces[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv.LINE_AA)
            cv.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        else:
            cv.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv.LINE_AA)
            cv.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
    
    cv.imshow('FRAME CAPTURADO', frame)

    k = cv.waitKey(1)
    if k == 27:
        break
    
vCap.release()
cv.destroyAllWindows()