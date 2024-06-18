import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt 
import pickle
from scipy.optimize import curve_fit

with open('ImagePts', 'rb') as fp:
     ImagePtsRead = pickle.load(fp)

with open('ObjPts', 'rb') as fp:
     ObjPtsRead = pickle.load(fp)
     
with open('CalibrationInfo', 'rb') as fp:
     CalibrationInfoRead = pickle.load(fp)

# Get obj to img transformation functions, draw grid
XObjExt = ObjPtsRead[0][0:21][:]
XImgExt = ImagePtsRead[0][0:21][:]
YObjExt = ObjPtsRead[0][21:31][:]
YObjExt = np.vstack([YObjExt, [0,0]])
YObjExt = np.vstack([YObjExt, ObjPtsRead[0][31:][:]])
YImgExt = ImagePtsRead[0][21:31][:]
YImgExt = np.vstack([YImgExt, XImgExt[10]])
YImgExt = np.vstack([YImgExt, ImagePtsRead[0][31:][:]])

Xx = XObjExt[:,0]
Yx = XImgExt[:,0]
Xy = YObjExt[:,1]
Yy = YImgExt[:,1]

nX = 2
nY = 2
Xstart = ((Yx[10]-Yx[9])+(Yx[11]-Yx[10]))/20
Xend = ((Yx[10]-Yx[0])+(Yx[20]-Yx[10]))/200
Ystart = ((Yy[10]-Yy[9])+(Yy[11]-Yy[10]))/20
Yend = ((Yy[10]-Yy[0])+(Yy[20]-Yy[10]))/200

def scaleX(X):
    return( Xstart + (pow((X/100),nX)*(Xend-Xstart)) )

def scaleY(Y):
    return( Ystart + (pow((Y/100),nY)*(Yend-Ystart)) )

sel_pts = ImagePtsRead[0]
CCx = CalibrationInfoRead[0][0]
CCy = CalibrationInfoRead[0][1]
fname = './ruler.jpg'
img = cv2.imread(fname)

scaled_pts = [[j for j in i] for i in sel_pts]
OriginOffset = [scaled_pts[10][0]-int(CCx), scaled_pts[10][1]-int(CCy)]
OLGimg = cv2.line(img, (int(CCx)+int(OriginOffset[0]), 0), (int(CCx)+int(OriginOffset[0]), img.shape[0]), (0,255,0), 1)
OLGimg = cv2.line(img, (0, int(CCy)+int(OriginOffset[1])), (img.shape[1], int(CCy)+int(OriginOffset[1])), (0,255,0), 1)


for i in range(101):
    if i % 10:
        OLGimg = cv2.line(OLGimg, (int(CCx)+int(OriginOffset[0])+int(i*scaleX(i)), 0), (int(CCx)+int(OriginOffset[0])+int(i*scaleX(i)), img.shape[0]), (0,255,0), 1)
        OLGimg = cv2.line(OLGimg, (0, int(CCy)+int(OriginOffset[1])+int(i*scaleY(i))), (img.shape[1], int(CCy)+int(OriginOffset[1])+int(i*scaleY(i))), (0,255,0), 1)
        OLGimg = cv2.line(OLGimg, (int(CCx)+int(OriginOffset[0])-int(i*scaleX(i)), 0), (int(CCx)+int(OriginOffset[0])-int(i*scaleX(i)), img.shape[0]), (0,255,0), 1)
        OLGimg = cv2.line(OLGimg, (0, int(CCy)+int(OriginOffset[1])-int(i*scaleY(i))), (img.shape[1], int(CCy)+int(OriginOffset[1])-int(i*scaleY(i))), (0,255,0), 1)
    if i % 10 == 0:
        OLGimg = cv2.line(OLGimg, (int(CCx)+int(OriginOffset[0])+int(i*scaleX(i)), 0), (int(CCx)+int(OriginOffset[0])+int(i*scaleX(i)), img.shape[0]), (255,0,0), 1)
        OLGimg = cv2.line(OLGimg, (0, int(CCy)+int(OriginOffset[1])+int(i*scaleY(i))), (img.shape[1], int(CCy)+int(OriginOffset[1])+int(i*scaleY(i))), (255,0,0), 1)
        OLGimg = cv2.line(OLGimg, (int(CCx)+int(OriginOffset[0])-int(i*scaleX(i)), 0), (int(CCx)+int(OriginOffset[0])-int(i*scaleX(i)), img.shape[0]), (255,0,0), 1)
        OLGimg = cv2.line(OLGimg, (0, int(CCy)+int(OriginOffset[1])-int(i*scaleY(i))), (img.shape[1], int(CCy)+int(OriginOffset[1])-int(i*scaleY(i))), (255,0,0), 1)

imgS = cv2.resize(OLGimg, (int(OLGimg.shape[1]/2), int(OLGimg.shape[0]/2)))
cv2.imshow('output',imgS)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('Grid.jpg', OLGimg)

CalibrationInfoWrite = []
CalibrationInfoWrite.append([nX, Xstart, Xend])
CalibrationInfoWrite.append([nY, Ystart, Yend])

with open('CalibrationFuncParams', 'wb') as fp:
    pickle.dump(CalibrationInfoWrite, fp)