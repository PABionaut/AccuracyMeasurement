# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:09:18 2024

@author: kevin
"""

import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt 
import pickle
import xlwt
import xlrd

import sympy as spy

def reg_sel_pt(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global ctr
        sel_pts.append([x+xoff,y+yoff])
        cv2.circle(imgS,(x,y),2,(0,0,255),-1)
        
def pix2XY(pixCo):
    X = spy.symbols ('X', real=True)
    Y = spy.symbols ('Y', real=True)
    xline = int(CalibrationInfoRead[0][1])
    yline = int(CalibrationInfoRead[0][0])
    pixX = pixCo[0]
    pixY = pixCo[1]
    EqX = spy.Eq( X*(Xstart + (pow((X/100),nX)*(Xend-Xstart))) + yline - pixX , 0)
    EqY = spy.Eq( Y*(Ystart + (pow((Y/100),nY)*(Yend-Ystart))) + xline - pixY , 0)
    
    xCo =  spy.solve(EqX)
    yCo =  spy.solve(EqY)
    
    return(float(xCo[0]), float(yCo[0]))

sel_pts = []
ctr = 0
CalibrationInfo = []
CalibrationImagePoints = []

with open('CalibrationInfo', 'rb') as fp:
     CalibrationInfoRead = pickle.load(fp)
with open('CalibrationFuncParams', 'rb') as fp:
     CalibrationFuncParamsRead = pickle.load(fp)
     
# Get fenestration point coordinates
nX = CalibrationFuncParamsRead[0][0]
nY = CalibrationFuncParamsRead[1][0]
Xstart = CalibrationFuncParamsRead[0][1]
Xend = CalibrationFuncParamsRead[0][2]
Ystart = CalibrationFuncParamsRead[1][1]
Yend = CalibrationFuncParamsRead[1][1]

book = xlrd.open_workbook("Samples.xls")
sh = book.sheet_by_index(0)

for i in range(0, sh.nrows):
    fname = r'C:\Users\pa480\OneDrive\Documents\GitHub\AccuracyMeasurement\IMAGES\CALIBRATION\\' + sh.cell_value(i,0) + '.jpg'
    img = cv2.imread(fname)
    
    cv2.namedWindow('Select Reference, Fenestration Point', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Select Reference, Fenestration Point', reg_sel_pt)
    
    xoff = 0
    yoff = 0
    panstep = 10
    while(1):
        if yoff<0:
            yoff = 0
        elif yoff>img.shape[1]:
            yoff = img.shape[1]-panstep
        if xoff<0:
            xoff = 0
        elif xoff>img.shape[0]:
            yoff = img.shape[1]-panstep
        imgS = img[yoff:,xoff:]
        cv2.imshow('Select Reference, Fenestration Point', imgS)
        key = cv2.waitKey(20)
        if key == 119:
            yoff = yoff-panstep
        elif key == 115:
            yoff = yoff+panstep
        elif key == 97:
            xoff = xoff-panstep
        elif key == 100:
            xoff = xoff+panstep
        elif cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

ref_pts_img = []
fen_pts_img = []

for j in range(0, sh.nrows*2):
    if j%2 == 0:
        ref_pts_img.append(sel_pts[j])
    elif j%2 == 1:
        fen_pts_img.append(sel_pts[j])
    
ref_pts_obj = []
fen_pts_obj = []
for j in range(0, sh.nrows):
    ref_pts_obj.append( [pix2XY(ref_pts_img[j])[0], pix2XY(ref_pts_img[j])[1]] )
    fen_pts_obj.append( [pix2XY(fen_pts_img[j])[0], pix2XY(fen_pts_img[j])[1]] )
    
ref_pts_obj = np.asarray(ref_pts_obj)
fen_pts_obj = np.asarray(fen_pts_obj)

# fenCo = pix2XY(scaled_pts[0])

# OLCHimg = cv2.line(OLCHimg, (CalibrationInfoRead[0][0],CalibrationInfoRead[0][1]), scaled_pts[0], (0,0,0), 2)
# OLCHimg = cv2.putText(OLCHimg, str(fenCo), [CalibrationInfoRead[0][0],CalibrationInfoRead[0][1]], cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

# cv2.imwrite('Annotated.jpg', OLCHimg)

excel_data = []
for i in range(0, sh.nrows):
    excel_data.append( [sh.cell_value(i,0), ref_pts_img[i], fen_pts_img[i], ref_pts_obj[i], fen_pts_obj[i], np.linalg.norm(fen_pts_obj[i]-ref_pts_obj[i])] )
    
wb = xlwt.Workbook()
ws = wb.add_sheet('measures')
style0 = xlwt.easyxf('font: name Times New Roman, color-index red, bold on',
    num_format_str='#,##0.00')
for i in range(0, sh.nrows):
    ws.write(i, 0, excel_data[i][0], style0)
    ws.write(i, 1, excel_data[i][1][0], style0)
    ws.write(i, 2, excel_data[i][1][1], style0)
    ws.write(i, 3, excel_data[i][2][0], style0)
    ws.write(i, 4, excel_data[i][2][1], style0)
    ws.write(i, 5, excel_data[i][3][0], style0)
    ws.write(i, 6, excel_data[i][3][1], style0)
    ws.write(i, 7, excel_data[i][4][0], style0)
    ws.write(i, 8, excel_data[i][4][1], style0)
    ws.write(i, 9, excel_data[i][5], style0)
    
wb.save('Accuracy Data.xls')