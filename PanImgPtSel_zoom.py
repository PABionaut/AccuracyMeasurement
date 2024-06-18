import cv2
import numpy as np
import pickle

sel_pts = []
ctr = 0
CalibrationInfo = []
CalibrationImagePoints = []

# Define the zoom factor
zoom_factor = 1.0

def on_mouse_event(event, x, y, flags, param):
    global sel_pts, ctr, zoom_factor, xoff, yoff, img, imgS, OLCHimg
    
    # Adjust mouse coordinates for zoom and offset
    x_adjusted = int((x) / zoom_factor)
    y_adjusted = int((y) / zoom_factor)
    # restart from here
    if event == cv2.EVENT_LBUTTONDBLCLK:
        sel_pts.append([int(((x) / zoom_factor)+xoff), int(((y) / zoom_factor)+yoff)])  # Store the coordinates of the clicked point
        cv2.circle(imgS, (x_adjusted, y_adjusted), 2, (0, 0, 255), -1)  # Draw a red dot at the clicked position
        
    if event == cv2.EVENT_MOUSEWHEEL:
        # Scroll up event
        if flags > 0:
            zoom_factor += 0.1  # Increase zoom factor
        # Scroll down event
        else:
            zoom_factor = max(0.1, zoom_factor - 0.1)  # Decrease zoom factor, but ensure it doesn't go below 0.1



fname = './ruler.jpg'
img = cv2.imread(fname)
img1 = cv2.imread(fname)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)

rows = gray.shape[0]

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=10, param2=30, minRadius=500, maxRadius=1200)

OLimg = cv2.circle(img, (int(circles[0, 0, 0]), int(circles[0, 0, 1])), int(circles[0, 0, 2]), (0, 0, 255), 1)
OLimgc1 = cv2.circle(img1, (int(circles[0, 0, 0]), int(circles[0, 0, 1])), int(circles[0, 0, 2]), (0, 0, 255), 1)
OLCHimg = cv2.line(OLimg, (int(circles[0, 0, 0]), 0), (int(circles[0, 0, 0]), img.shape[0]), (0, 0, 255), 3)
OLCHimg = cv2.line(OLCHimg, (0, int(circles[0, 0, 1])), (img.shape[1], int(circles[0, 0, 1])), (0, 0, 255), 3)
OLCHimg = cv2.line(OLimg, (int(img.shape[1] / 2), 0), (int(img.shape[1] / 2), img.shape[0]), (0, 255, 0), 3)
OLCHimg = cv2.line(OLCHimg, (0, int(img.shape[0] / 2)), (img.shape[1], int(img.shape[0] / 2)), (0, 255, 0), 3)

cv2.namedWindow('Select Scale Points', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('Select Scale Points', on_mouse_event)

xoff = 0
yoff = 0
panstep = 10
zoom_factor = 1.0

while True:
    
    if yoff < 0:
        yoff = 0
    elif yoff > img.shape[1]:
        yoff = img.shape[1] - panstep
    if xoff < 0:
        xoff = 0
    elif xoff > img.shape[0]:
        yoff = img.shape[1] - panstep
    
    imgS = OLCHimg[yoff:, xoff:]

    #imgS_resized = cv2.resize(imgS, (int(imgS.shape[1] * zoom_factor), int(imgS.shape[0] * zoom_factor)))
    
    cv2.resize
    key = cv2.waitKey(20)
    if key == 119:  # 'w'
        yoff -= panstep
    elif key == 115:  # 's'
        yoff += panstep
    elif key == 97:  # 'a'
        xoff -= panstep
    elif key == 100:  # 'd'
        xoff += panstep
    elif key & 0xFF == 27:  # Escape key
        break
    
    imgS_resized = cv2.resize(imgS, None, fx=zoom_factor, fy=zoom_factor)
    #print(zoom_factor)
    # Display the resized image
    cv2.imshow('Select Scale Points', imgS_resized)
    
    # Call the mouse event handler with adjusted coordinates
    cv2.setMouseCallback('Select Scale Points', on_mouse_event)

cv2.destroyAllWindows()


img_pts = np.zeros(shape=(41,2))
# IndexError: List index out of range
for ctr in range (0,41):
    if ctr < 21:
        img_pts[ctr,:] = [sel_pts[ctr][0], int(circles[0,0,1])]
    if ctr > 20 and ctr < 42:
        img_pts[ctr,:] = [int(circles[0,0,0]), sel_pts[ctr][1]]
    if len(sel_pts) < 41:
        print("Error: not enough points selected. Please select 41 points.")
        break  # Or continue to let the user know they need to select more points
    


img_pts32 = [np.float32(img_pts)]
        
obj_pts = np.zeros(shape=(41,2))
obj_pts[0:21,0] = list(range(-10,11))
obj_pts[0:21,1] = 0
obj_pts[21:31, 1] = list(range(10,0,-1))
obj_pts[21:, 0] = 0
obj_pts[31:, 1] = list(range(-1,-11,-1))

obj_pts32 = [np.float32(obj_pts)]

CalibrationInfo.append([int(circles[0,0,0]), int(circles[0,0,1]), int(circles[0,0,2])])

with open('CalibrationInfo', 'wb') as fp:
    pickle.dump(CalibrationInfo, fp)
    
with open('ImagePts', 'wb') as fp:
    pickle.dump(img_pts32, fp)
    
with open('ObjPts', 'wb') as fp:
    pickle.dump(obj_pts32, fp)