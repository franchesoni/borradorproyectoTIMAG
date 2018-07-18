#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 16:31:55 2018

@author: nicolas
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from alinear_UTE import align
plt.close('all')
#%%
#Carga de archivos
regions = []
binary = []
for i in [3, 4, 5, 6, 7, 8, 9, 10, 11]:
    filename = '../../images/regiones_UTE/region_ute{}.jpg'.format(i)
    img = cv2.imread(filename)
    regions.append(img)
del img
kernel = np.ones((2,2),np.uint8)
for index, i in enumerate([3, 4, 5, 6, 7, 8, 9, 10, 11]):
    binary.append(cv2.cvtColor(regions[index], cv2.COLOR_BGR2GRAY))
    th3 = cv2.adaptiveThreshold(binary[-1],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,5)
    binary[-1] = cv2.morphologyEx(th3, cv2.MORPH_ERODE, kernel)
    binary[-1] = cv2.medianBlur(binary[-1], 11)
    cv2.imwrite('../../images/regiones_UTE/binary{}.jpg'.format(i), binary[-1])