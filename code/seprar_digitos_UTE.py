#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 16:31:55 2018

@author: nicolas
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from alinear_UTE import alinear_region_UTE

#%%
#%%
#Carga de archivos
regions = []
for i in [3, 4, 5, 6, 7, 8, 9, 10, 11]:
    filename = '../images/regiones_UTE/region_ute{}.jpg'.format(i)
    img = cv2.imread(filename)
    regions.append(img)
    
del filename
del img
del i
#%%
img = cv2.cvtColor(regions[6], cv2.COLOR_BGR2GRAY)
plt.figure(); plt.imshow(img)

#%%

th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,5)
plt.figure(); plt.imshow(th3)

kernel = np.ones((2,2),np.uint8)
img2 = cv2.morphologyEx(th3, cv2.MORPH_ERODE, kernel)
plt.figure(); plt.imshow(img2)
#%%
img3 = cv2.medianBlur(img2, 11)
plt.figure(); plt.imshow(img3)