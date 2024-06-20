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
from matplotlib.backend_bases import MouseButton

matplotlib.use('Qt5Agg')  # Use Qt5Agg for GUI handling, adjust if necessary

'''
def correctionFactor(x):
    if x <= 836:
        calX = int(x-.0588*x-0.3196)
        print('X: ' + str(x))
        print('Cal X: ' + str(calX))
        return calX
    if x > 336:
        return x
'''

class RectangleSelector:
    def __init__(self, ax, img):
        self.ax = ax
        self.img = img
        self.rectangles = []
        self.centroid_markers = []
        self.centroid_coordinates = []

        self.RS = widgets.RectangleSelector(
            self.ax, self.on_select,
            useblit=True,
            button=[MouseButton.LEFT],  # Left mouse button
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )

        # Bind zoom functionality
        self.zoom_id = self.ax.figure.canvas.mpl_connect('scroll_event', self.zoom)

    def on_select(self, eclick, erelease):
        x1, y1, x2, y2 = int(eclick.xdata), int(eclick.ydata), int(erelease.xdata), int(erelease.ydata)

        mask = np.zeros(self.img.shape, dtype=bool)
        mask[y1:y2, x1:x2] = True
        selected_region = np.where(mask & (self.img == 0))
        centroid_x = int(np.mean(selected_region[1]))
        centroid_y = int(np.mean(selected_region[0]))
        self.centroid_coordinates.append((centroid_x, centroid_y))

        #newX = correctionFactor(centroid_x)
        sel_pts.append([centroid_x, centroid_y])

        print(f"Centroid {len(self.centroid_markers) + 1}: ({centroid_x}, {centroid_y})")

        color = 'red' if len(self.centroid_markers) % 2 == 0 else 'blue'
        centroid_marker = self.ax.scatter(centroid_x, centroid_y, color=color, s=20, zorder=5)
        self.centroid_markers.append(centroid_marker)

        rectangle = self.ax.add_patch(
            plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                          edgecolor=color, facecolor='none', linewidth=2)
        )
        self.rectangles.append(rectangle)

        if len(self.centroid_coordinates) == 2:
            line = plt.Line2D(*zip(*self.centroid_coordinates), color='orange', linestyle='dotted', linewidth=2)
            self.ax.add_line(line)

        self.ax.figure.canvas.draw()

    def zoom(self, event):
        base_scale = 1.1
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        
        xdata = event.xdata
        ydata = event.ydata

        if xdata is None or ydata is None:
            return

        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            scale_factor = 1

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        self.ax.figure.canvas.draw()
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
    
    # Load the uploaded image
    img_path = r'C:\Users\pa480\OneDrive\Documents\GitHub\AccuracyMeasurement\{}'.format(fname)
    img = Image.open(img_path)
    print('image name: '+str(sh.cell_value(i,0)))
    # Convert to grayscale
    gray_img = img.convert('L')  # 'L' for grayscale

    # Apply a custom threshold and convert directly to binary format
    threshold = 110  # Define your own threshold here, range is 0 to 255
    bw_custom_threshold_img = gray_img.point(lambda x: 255 if x > threshold else 0, mode='1')

    # Save the black and white image with custom threshold
    bw_custom_threshold_img_path = r'C:\Users\pa480\OneDrive\Documents\GitHub\AccuracyMeasurement\{}_BW.jpg'.format(sh.cell_value(i,0))
    bw_custom_threshold_img.save(bw_custom_threshold_img_path)

    #w_custom_threshold_img.show()  # This line is to display the image if running locally
    #bw_custom_threshold_img_path  # This would usually print the path or return it in a script

    # Load the uploaded image
    img_path = r'C:\Users\pa480\OneDrive\Documents\GitHub\AccuracyMeasurement\{}_BW.jpg'.format(sh.cell_value(i,0))
    img = Image.open(img_path).convert("L")

    # Convert to numpy array
    img_array = np.array(img)
    
    # Create a figure and axis to plot on
    fig, ax = plt.subplots()
    ax.imshow(img_array, cmap='gray')
    ax.set_title("Draw rectangles around the objects and view centroid coordinates and distance in console")
    ax.axis('off')


    # Create an instance of the RectangleSelector
    selector = RectangleSelector(ax, img_array)
    
    plt.show(block=False)
    while plt.get_fignums():  # Loop until all figures have been closed
        plt.pause(0.5)  # Wait briefly for GUI events

    
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