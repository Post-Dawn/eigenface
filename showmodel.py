import cv2
#import argparse
import sys
import numpy as np
import tool

if(len(sys.argv)==2):
    model_path=sys.argv[1]
else:
    print("ERROR: There is enough information to continue.")
    exit()

try:
    w, v = tool.getModel(model_path)
except:
    print("ERROR: There is no model file.")
    exit()

#print(w)
n, m = v.shape
print("n = " + str(n) + ", m = " + str(m))

eigen_face = np.zeros((tool.Height, tool.Width))
for i in range(m):
    img = np.resize(v[:, i], (tool.Height, tool.Width))
    #print(img)
    eigen_face += (img + 1) * w[i]
eigen_face = cv2.equalizeHist(((eigen_face / eigen_face.max()) * 255).astype(np.uint8))
cv2.imshow("eigen face", eigen_face)

cv2.waitKey(0)