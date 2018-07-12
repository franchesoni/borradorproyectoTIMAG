#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 15:49:49 2018

@author: franchesoni
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter


#%% IMAGES LOAD
img = cv2.imread('../../images/OSE/ose6.jpg')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
edges = np.zeros_like(rgb)
#%%
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue = hsv[:,:,0]
ret, thresh = cv2.threshold(hue, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#%% TRY TO FIND CONTOURS
(_, contours, _) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
largo = 1000
for contour in contours:
    if contour.size > largo:
        cv2.drawContours(edges, [contour], -1, (255, 255, 255), 3)
edges = edges[:,:,0]
#%%
result = hough_ellipse(edges, accuracy=20, threshold=250,
                       min_size=100, max_size=120)     
#%%
#plt.close('all')
#plt.figure()
#plt.imshow(edges)