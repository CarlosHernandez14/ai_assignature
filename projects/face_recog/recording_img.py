
import cv2 as cv
import numpy as np


vCap = cv.VideoCapture(0);
faceDetector = cv.CascadeClassifier('../haarcascade_frontalface_alt.xml')

x=y=w=h=0
img: np.ndarray = np.zeros((1, 1, 3), dtype=np.uint8)
count = 0

labelI = 916
while True:
    ret, frame = vCap.read()
    if not ret:
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # cvtColor, scaleFacto, neightboursCount
    faces = faceDetector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces: 
        m = int(h/2)

        faceFound = frame[y: y + h, x: x + w]
        faceFound = cv.resize(faceFound, (100, 100), interpolation=cv.INTER_AREA)

        if (labelI % 1 == 0):
            cv.imwrite('/Users/carloskvpchc/Documents/tec_projects/ai_assignature/projects/face_recog/datasets/dalia/dalia'+str(labelI)+'.jpg', faceFound)
            cv.imshow('ROSTRO DETECTADO', faceFound)


        roi = frame[y:y + h, x:x + w]
        img = cv.subtract(np.full_like(roi, 180), roi)
        count += 1

    cv.imshow('FACES', frame)
    labelI += 1
    k = cv.waitKey(1)
    if k == 27:
        break


vCap.release()
cv.destroyAllWindows()
