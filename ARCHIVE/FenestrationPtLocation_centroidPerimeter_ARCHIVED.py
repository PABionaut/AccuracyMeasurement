'''
#######################################
### pip install numpy pillow matplotlib
#######################################
'''

import cv2
import numpy as np
import os
import glob
import pickle
import xlwt
import xlrd
import sympy as spy
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import time

matplotlib.use('Qt5AGG')  # Use TKAgg for GUI handling, change if necessary
# switch to 'Qt5Agg' or another backend that is appropriate for the system

def correctionFactor(x):
    if x <= 836:
        calX = int(x-.0588*x-0.3196)
        print('X: ' + str(x))
        print('Cal X: ' + str(calX))
        return calX
    if x > 336:
        return x
    
def click_event(event, x, y, flags, params):
    global points, img, centroids
    
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)  # Mark the point of click
        points.append((x, y))
        
        if len(points) == 3:
            center, radius = calculate_circle(points)
            if center is not None and radius > 0:
                cv2.circle(img, center, radius, (255, 0, 0), 2)  # Draw the circle
                cv2.circle(img, center, 5, (0, 0, 255), -1)  # Mark the centroid
                centroids.append(center)
                print(centroids)
                if len(centroids) == 2:
                    draw_dotted_line(centroids[0], centroids[1])
            points.clear()  # Reset points for next circle
        
        
        cv2.imshow('image', img)

def calculate_circle(pts):
    x1, y1 = pts[0]
    x2, y2 = pts[1]
    x3, y3 = pts[2]

    A = np.array([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]])
    B = np.array([[-x1**2 - y1**2], [-x2**2 - y2**2], [-x3**2 - y3**2]])

    if np.linalg.det(A) == 0:
        return (None, None)  # Cannot solve if the matrix is singular

    sol = np.linalg.solve(A, B)
    a, b, c = sol.flatten()
    x0, y0 = -a / 2, -b / 2
    r = np.sqrt(x0**2 + y0**2 - c)

    return (int(x0), int(y0)), int(r)

def draw_dotted_line(pt1, pt2):
    
    # Function to draw a dotted line between two points
    color = (0, 255, 255)  # Cyan color for the dotted line
    line_type = cv2.LINE_AA  # Anti-aliased line for better visualization
    thickness = 2
    gap = 20  # Gap between dots
    dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
    print(f"Distance between centroids: {dist:.2f} pixels")

    for i in np.arange(0, 1, 1 / (dist / gap)):
        r = i * np.array(pt2) + (1 - i) * np.array(pt1)
        cv2.circle(img, (int(r[0]), int(r[1])), thickness, color, -1, line_type)
        
        

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
    fname = './' + sh.cell_value(i,0) + '.jpg'
    
    # Setup
    img = cv2.imread(r'C:\Users\pa480\OneDrive\Documents\Bionaut\Contract work\Fenestration tutorial\{}'.format(fname))  # Load the image
    points = []  # List to store points
    centroids = []  # List to store centroids
    
    cv2.namedWindow('image')  # Create a window named 'image'
    cv2.setMouseCallback('image', click_event)  # Set mouse callback
    
    cv2.imshow('image', img)
    cv2.waitKey(0)
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