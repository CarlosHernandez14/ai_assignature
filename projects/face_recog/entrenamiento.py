import os
import time
import cv2 as cv
import numpy as np

# Rutas
dataset = '/Users/carloskvpchc/Documents/tec_projects/ai_assignature/projects/face_recog/datasets'
pathModels = '/Users/carloskvpchc/Documents/tec_projects/ai_assignature/projects/face_recog/trained_models'
os.makedirs(pathModels, exist_ok=True)

faces  = os.listdir(dataset)
print(faces)

labels = []
facesData = []
label = 0 
for face in faces:
    facePath = dataset+'/'+face
    for faceName in os.listdir(facePath):
        img = cv.imread(facePath+'/'+faceName, 0)
        if img is None:
            continue
        img = cv.resize(img, (48, 48), interpolation=cv.INTER_AREA)
        facesData.append(img)
        labels.append(label)
    label = label + 1

faceRecognizer = cv.face.FisherFaceRecognizer_create() # type: ignore
faceRecognizer.train(facesData, np.array(labels))
faceRecognizer.write(os.path.join(pathModels, 'FisherFace.xml'))

