import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image, ImageOps, ImageEnhance, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

def upload_image():
    # Open a file dialog to select an image file
    file_path = r'C:\Users\pa480\OneDrive\Documents\GitHub\AccuracyMeasurement\calibration_cross_hair.jpg'
    if file_path:
        process_image(file_path)

def process_image(file_path):
    # Open the selected image
    image = Image.open(file_path)

    # Convert the image to grayscale
    grayscale_image = ImageOps.grayscale(image)

    # Enhance contrast (for better thresholding effect)
    enhancer = ImageEnhance.Contrast(grayscale_image)
    enhanced_image = enhancer.enhance(2)

    # Ask for threshold value
    threshold = 125

    # Apply the threshold
    bw_image = enhanced_image.point(lambda x: 255 if x > threshold else 0, mode='1')

    # Draw a circle based on user click
    draw_circle(bw_image)

def draw_circle(image):
    # Convert to RGB to draw colored circles
    image_with_circle = image.convert("RGB")
    draw = ImageDraw.Draw(image_with_circle)

    # Initial zoom level
    zoom_level = 1.0
    center = [image_with_circle.width / 2, image_with_circle.height / 2]

    # Display the image and allow the user to click a point
    fig, ax = plt.subplots()
    ax.imshow(image_with_circle)

    def onscroll(event):
        nonlocal zoom_level
        # Adjust zoom level
        if event.button == 'up':
            zoom_level *= 1.1  # Zoom in
        elif event.button == 'down':
            zoom_level /= 1.1  # Zoom out
        # Redraw the image with the updated zoom level
        redraw_image()

    def redraw_image():
        ax.clear()
        ax.imshow(image_with_circle)
        # Set new limits for x and y axis based on zoom level and center position
        ax.set_xlim([center[0] - (image_with_circle.width / zoom_level) / 2,
                     center[0] + (image_with_circle.width / zoom_level) / 2])
        ax.set_ylim([center[1] + (image_with_circle.height / zoom_level) / 2,
                     center[1] - (image_with_circle.height / zoom_level) / 2])  # Reversed to maintain orientation
        fig.canvas.draw()

    def onclick(event):
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            # Define the radius of the circle
            radius = 700

            # Draw a circle at the clicked point
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline="black", width=3)
            mask_image_outside_circle(image_with_circle, x, y, radius)
            redraw_image()
            fig.canvas.mpl_disconnect(cid_click)  # Disable further clicks
            save_processed_image(image_with_circle)

    cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('scroll_event', onscroll)

    plt.show()

def mask_image_outside_circle(image, center_x, center_y, radius):
    # Create a mask image with the same size as the original image
    mask = Image.new("L", image.size, 0)

    # Draw a white circle on the mask where we want to keep the original image
    draw = ImageDraw.Draw(mask)
    draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), fill=255)

    # Create a white image for the background
    white_bg = Image.new("RGB", image.size, "white")

    # Composite the image with the white background using the mask
    image.paste(white_bg, (0, 0), mask=ImageOps.invert(mask))

def save_processed_image(image_with_circle):
    # Save the image with the circle and masked area
    save_path = r'C:\Users\pa480\OneDrive\Documents\GitHub\AccuracyMeasurement\ruler.jpg'
    if save_path:
        image_with_circle.save(save_path)
        print(f"Image saved to {save_path}")

if __name__ == "__main__":
    # Create a Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Upload and process the image
    upload_image()
