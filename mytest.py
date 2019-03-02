import cv2
import numpy as np
import os
import sys
import tool

def calVec(classifier, v, img):
    face_rects = classifier.detectMultiScale(img, 1.3, 5)
    x, y, w, h = face_rects[0]
    imgNow = cv2.equalizeHist(img[y: y + h, x: x + w])
    imgNow = cv2.resize(imgNow, (tool.Width, tool.Height), interpolation = cv2.INTER_AREA)
    vec=imgNow.flatten()
    return vec

def difference(classifier, v, a, b):
    x = np.dot(v.T, calVec(classifier, v, a))
    y = np.dot(v.T, calVec(classifier, v, b))
    return np.linalg.norm(x - y)

def predict(classifier, v, img, train_set):
    minVal = None
    ans = None
    for item in train_set:
        fileName=item[0]
        fileImg=item[1]
        diff = difference(classifier, v, img, fileImg)
        if ans==None or diff < minVal:
            ans = (fileName, fileImg)
            minVal = diff
    return ans


if __name__ == '__main__':
    model_path="model"
    if(len(sys.argv)==2):
        model_path=sys.argv[1]

    print("model_File = "+model_path)
    try:
        w, v = tool.getModel(model_path)
    except:
        print("ERROR: model file not exists.")
        exit()

    classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    source="train"
    train_set = []
    for fileName in os.listdir(source):
        fileArray=fileName.split(".")
        if not fileArray[-1]=="tiff": continue
        print(fileName)
        img = cv2.imread(source+"/" + fileName, cv2.IMREAD_GRAYSCALE)
        train_set.append((source+"/" + fileName, img))

    print("The image to be predicted: ")
    img_path = input()
    print('image path =', img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Can't find image!")
        exit()

    result = predict(classifier, v, img, train_set)
    print("Result image name =", result[0])
    cv2.imshow('exam image', img)
    cv2.imshow('result image', result[1])
    cv2.waitKey(0)