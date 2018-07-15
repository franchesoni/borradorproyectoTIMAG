#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 18:59:00 2018

@author: franchesoni
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from detect import detect

plt.close('all')
# en las que funcionan
for i in [3, 4, 5, 6, 7, 8, 9, 10, 11]:
    #Carga de archivos
    filename = '../../images/UTE/ute{}.jpg'.format(i)
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 55)
    #%%
    #Aplico umbral para tener imagen binaria
    ret, thresh = cv2.threshold(blurred, 127, 255, 0)
    #Hallo contornos
    (_, contours, _) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #Hallo contornos que sean rectangulos y los ploteo
    rectangles = []
    for contour in contours:
        shape = detect(contour)
        if shape == "rectangle":
            rect = cv2.boundingRect(contour)
            rectangles.append(rect)
    
    total_area = thresh.size
    max_allowed = total_area*0.3
    rectangles = [rectangle for rectangle in rectangles if rectangle[2] * rectangle[3] < max_allowed]
    ganamos = np.argmax([rectangle[2] * rectangle[3] for rectangle in rectangles])
    
    rect = rectangles[ganamos]
    region = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2], :]
    cv2.imwrite('../../images/regiones_UTE/region_ute{}.jpg'.format(i), region)
