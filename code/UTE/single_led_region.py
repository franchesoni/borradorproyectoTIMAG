#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 09:58:47 2018

@author: franchesoni
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from detect import detect

# This script takes the picture of an electric meter and shows the detected
# area for the LED screen. The steps are
# 1 - Load and treatment of image
# 2 - Apply threshold and find contours
# 3 - Get contours of rectangular shape
# 4 - Select the one with greater area (but not more than a third of the image)
# 5 - Display the region detected
#%%
#Carga de archivos
filename = '../../images/UTE/ute5.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.medianBlur(gray, 55)
#%%
#Aplico umbral para tener imagen binaria
ret,thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#Hallo contornos
(_, contours, _) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#Hallo contornos que sean rectangulos y los ploteo
rectangles = []
for contour in contours:
    shape = detect(contour)
    if shape == "rectangle":
        rect = cv2.boundingRect(contour)
        rectangles.append(rect)
        # Add contours and rectangles to image
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 3)
        cv2.rectangle(img, (rect[0],rect[1]), (rect[2]+rect[0],rect[3]+rect[1]), (255,0,0), 3)
#%%
# Select that with area constrains 
total_area = thresh.size
max_allowed = total_area*0.3
rectangles = [rectangle for rectangle in rectangles if rectangle[2] * rectangle[3] < max_allowed]
selected_index = np.argmax([rectangle[2] * rectangle[3] for rectangle in rectangles])
rect = rectangles[selected_index]
#%%  Display
cv2.rectangle(img, (rect[0],rect[1]), (rect[2]+rect[0],rect[3]+rect[1]), (0,0,255), 20)
plt.close('all')
plt.figure()
plt.imshow(img)