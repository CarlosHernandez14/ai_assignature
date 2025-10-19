
import cv2 as cv
import numpy as np
import time

modelPath = '/Users/carloskvpchc/Documents/tec_projects/ai_assignature/projects/expression_recog/trained_models/FisherFace.xml'
haarcascade = '/Users/carloskvpchc/Documents/tec_projects/ai_assignature/haarcascade_frontalface_alt.xml'


faces = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

vCap = cv.VideoCapture(0)
faceClasifier = cv.CascadeClassifier(haarcascade)



faceRecognizer = cv.face.FisherFaceRecognizer_create()  # type: ignore[attr-defined]
faceRecognizer.read(modelPath)

while True: 
    # Capture frames
    ret, frame = vCap.read()
    if ret == False : break

    # Color classifier
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    grayCopy = gray.copy()

    facesDetected = faceClasifier.detectMultiScale(gray, 1.3, 3)
    # Iterate over the detected faces in the frame
    for (x, y , w, h)in facesDetected:
        
        frameGray = grayCopy[y: y + h, x: x + w]
        frameGray = cv.resize(frameGray, (48, 48), interpolation=cv.INTER_CUBIC)
        
        result = faceRecognizer.predict(frameGray)
        cv.putText(frame, '{}'.format(result), (x,y-20), 1,3.3, (255,255,0), 1, cv.LINE_AA)
        if result[1] < 500:
            cv.putText(frame,'{}'.format(faces[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv.LINE_AA)
            cv.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        else:
            cv.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv.LINE_AA)
            cv.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
    
    cv.imshow('FRAME', frame)

    k = cv.waitKey(1)
    if k == 27:
        break

print('Import Error')
vCap.release()
cv.destroyAllWindows()