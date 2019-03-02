import os
import cv2
import random
import sys
import scipy.linalg
import numpy
import tool

if __name__ == '__main__':
    k = 100# number of eigen values and their corresponding eigen vectors
    if(len(sys.argv)<=2):
        print("ERROR: do not have valid and enough information to continue.")
        exit()
    elif(len(sys.argv)==3):
        percent=int(sys.argv[1])
        model_path=sys.argv[2]
    else:
        percent=int(sys.argv[1])
        model_path=sys.argv[2]
        k=sys.argv[3]

    source="train"
    print('Loading training set')
    img = []
    fileList=[]
    for fileName in os.listdir(source):
        fileArray=fileName.split(".")
        if not fileArray[-1]=="tiff": continue
        fileList.append(fileName)
    
    lenFile=len(fileList)
    deleteNum=int(lenFile*(100-percent)/100)
    for i in range(deleteNum):
        posDel=random.randint(0,lenFile-i-1)
        del fileList[posDel]

    count=0
    for fileName in fileList:
        print(fileName)    
        count=count+1
        img.append(cv2.imread(source+'/' + fileName, cv2.IMREAD_GRAYSCALE))

    print("Total file number = "+str(count))
    
    print('Loading classifier model')
    classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    imgMatrix = tool.imgToMatrix(classifier, img)

    print('Calculating covariance matrix')
    cov = numpy.cov(imgMatrix.T)
    print('Finding eigen')
    w, v = scipy.linalg.eigh(cov, eigvals = (tool.Sum - k, tool.Sum - 1))
    print('Writing to', model_path)
    numpy.savetxt(model_path, numpy.vstack([v,w]), delimiter = ',')