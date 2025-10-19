import cv2 as cv
import os
import time
import numpy as np

dataset = '/Users/carloskvpchc/Documents/tec_projects/ai_assignature/projects/expression_recog/images/train'
pathModels = '/Users/carloskvpchc/Documents/tec_projects/ai_assignature/projects/expression_recog/trained_models'

faceFeelings = os.listdir(dataset)
print(faceFeelings)

labels = []
feelingsData = []

lbl = 0

for expresssion in faceFeelings:
    expPath = f'{dataset}/{expresssion}'

    for exprName in os.listdir(expPath):
        img = cv.imread(f'{expPath}/{exprName}', 0)

        if img is None:
            continue
        
        feelingsData.append(img)
        labels.append(lbl)

    lbl += 1

faceRecog = cv.face.FisherFaceRecognizer_create()  # type: ignore
faceRecog.train(feelingsData, np.array(labels))
faceRecog.write(os.path.join(pathModels, 'FisherFace.xml'))
