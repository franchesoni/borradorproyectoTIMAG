#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 12:07:33 2018

@author: franchesoni
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageFilter

images = []
#pil_images = []
for i in range(3, 12):
    images.append(cv2.imread('../images/regiones_UTE/region_ute{}.jpg'.format(i)))
#    pil_images.append(Image.open('../images/regiones_UTE/region_ute{}.jpg'.format(i)))
    
blurred_images = []
thresholdings = [] 
filtered = []
for img in images:
    blurred_images.append(cv2.medianBlur(img, 7))
    thresholdings.append(
           cv2.threshold(cv2.cvtColor(blurred_images[-1], cv2.COLOR_BGR2GRAY), 
                         0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1])
    filtered.append(cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21))

#for pilimg in pil_images:    
#    filtered.append(pilimg.filter(ImageFilter.SHARPEN))

plt.close('all')
for i in range(9):
    plt.figure()
    plt.subplot(221)
    plt.imshow(images[i])
    plt.subplot(222)
    plt.imshow(blurred_images[i])
    plt.subplot(223)
    plt.imshow(thresholdings[i])
    plt.subplot(224)
    plt.imshow(filtered[i])
    
#plt.close('all')
#plt.figure(figsize=(20, 20))
#plt.subplot(331); plt.imshow(img)
#plt.subplot(332); plt.imshow(blurred)


