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
for i in [3, 4, 5, 8, 9, 11]:
    #Carga de archivos
    filename = '../images/UTE/ute{}.jpg'.format(i)
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #%% modification
    blurred = cv2.blur(gray, (10, 10))
    # se rompe, por que?
#    blurred = cv2.medianBlur(gray, 55)
    #%%
    #Aplico umbral para tener imagen binaria
    ret, thresh = cv2.threshold(blurred, 127, 255, 0)
    #Hallo contornos
    (_, contours, _) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #Hallo contornos que sean rectangulos y los ploteo
    rectangles = []
    for contour in contours:
        M = cv2.moments(contour)
        shape = detect(contour)
        if shape == "rectangle":
            cv2.drawContours(img, [contour], -1, (0, 255, 0), 3)
            rect = cv2.boundingRect(contour)
            rectangles.append(rect)
            cv2.rectangle(img, (rect[0],rect[1]), (rect[2]+rect[0],rect[3]+rect[1]), (255,0,0), 3)
    
    total_area = thresh.size
    max_allowed = total_area*0.3
    
    ganamos = np.argmax(np.array([rectangle[2] * rectangle[3] for rectangle in rectangles if rectangle[2] * rectangle[3] < max_allowed]))
    
    rect = rectangles[ganamos]
    cv2.rectangle(img, (rect[0],rect[1]), (rect[2]+rect[0],rect[3]+rect[1]), (0,0,255), 20)
    
    plt.figure()
    plt.imshow(img)