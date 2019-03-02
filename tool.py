import cv2
import numpy as np

Width = 80
Height = 80
Sum=Width*Height

def imgToMatrix(classifier, imgs):
    n=len(imgs)
    imgMatrix = np.zeros((n, Sum))
    for i in range(n):
        imgNow=imgs[i]
        face_rects = classifier.detectMultiScale(imgNow, 1.3, 5)
        x, y, w, h = face_rects[0]
        imgNow = cv2.equalizeHist(imgNow[y: y + h, x: x + w])
        imgNow = cv2.resize(imgNow, (Width, Height), interpolation = cv2.INTER_AREA)
        
        imgVector=imgNow.flatten()
        imgMatrix[i]=imgVector
    return imgMatrix     

def getModel(path):
    model = np.genfromtxt(path, delimiter = ',')
    w = model[-1]
    v = model[0:-1]
    return (w, v)