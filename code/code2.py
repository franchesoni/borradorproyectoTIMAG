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

plt.close('all')
#Carga de archivos
filename = '../images/UTE/ute12.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.medianBlur(gray, 55)
#%%
#Aplico umbral para tener imagen binaria
ret,thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#ret, thresh = cv2.threshold(blurred, 127, 255, 0)
#Hallo contornos
plt.figure()
plt.imshow(thresh)

(_, contours, _) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#Hallo contornos que sean rectangulos y los ploteo
rectangles = []
for contour in contours:
    shape = detect(contour)
    if shape == "rectangle":
#        peri = cv2.arcLength(contour, True)
#        approx = cv2.approxPolyDP(contour, 0.08 * peri, True)
#        cv2.drawContours(img, [approx], -1, (255, 0, 255), 3)
    
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 3)
        rect = cv2.boundingRect(contour)
        rectangles.append(rect)
        cv2.rectangle(img, (rect[0],rect[1]), (rect[2]+rect[0],rect[3]+rect[1]), (255,0,0), 3)

total_area = thresh.size
max_allowed = total_area*0.3

rectangles = [rectangle for rectangle in rectangles if rectangle[2] * rectangle[3] < max_allowed]
ganamos = np.argmax([rectangle[2] * rectangle[3] for rectangle in rectangles])

rect = rectangles[ganamos]
cv2.rectangle(img, (rect[0],rect[1]), (rect[2]+rect[0],rect[3]+rect[1]), (0,0,255), 20)

plt.figure()
plt.imshow(img)