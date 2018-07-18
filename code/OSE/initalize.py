#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 14:31:13 2018

@author: franchesoni
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
plt.rcParams['image.cmap'] = 'gray'

#%% IMAGES LOAD
img = cv2.imread('../images/OSE/ose12b.jpg')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
nothing = np.zeros_like(rgb)
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
        cv2.drawContours(nothing, [contour], -1, (255, 255, 255), 3)
#%% DISPLAY
plt.close('all')
plt.imshow(nothing)
#plt.figure()
#plt.imshow(thresh)