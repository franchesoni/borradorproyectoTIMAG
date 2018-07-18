#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 12:07:33 2018

@author: franchesoni
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


plt.close('all')
images = []
for i in range(3, 12):
    images.append(cv2.imread('../../images/regiones_UTE/binary{}.jpg'.format(i)))
templates = []
for i in range(1, 4):
    templates.append((cv2.imread('../../images/Extras/template{}.jpg'.format(i)))[55:260, 40:310])
    plt.figure()
    plt.imshow(templates[-1])
    
img = images[0]
ar = templates[0].shape[0]/templates[0].shape[1]
tmp = templates[0]


#img_fft = np.fft.fft2(img)
#tmp_fft = np.fft.fft2(tmp)
#result = np.fft.ifft2(img_fft*tmp_fft)
plt.figure()
plt.imshow(img)

# tenemos 3 templates
# iteramos sobre los templates por la imagen con phase correlation
# tambien cambiamos el tamanio del template
