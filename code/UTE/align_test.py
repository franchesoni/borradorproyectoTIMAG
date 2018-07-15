#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 18:50:58 2018

@author: franchesoni
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from alinear_UTE import align

region = cv2.imread('../../images/regiones_UTE/region_ute5.jpg')
gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
ret,th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kernel = np.ones((5,5),np.uint8)
close = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel) 
kernel2 = np.ones((5,5),np.uint8)
th = cv2.morphologyEx(close, cv2.MORPH_DILATE, kernel2) 
ptos = np.argwhere(th < 250)
suma = np.sum(ptos, axis=1) 
resta = np.diff(ptos, axis=1)

A = ptos[np.argmin(suma)]
B = ptos[np.argmin(resta)]
C = ptos[np.argmax(suma)]
D = ptos[np.argmax(resta)]

plt.close('all')
plt.figure()
plt.imshow(th)
plt.plot(A[1], A[0], 'rx')
plt.plot(B[1], B[0], 'rx')
plt.plot(C[1], C[0], 'rx')
plt.plot(D[1], D[0], 'rx')

#%%

alineada = align(region)
plt.figure()
plt.imshow(region)
plt.figure()
plt.imshow(alineada)