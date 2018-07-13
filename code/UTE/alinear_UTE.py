#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 14:43:35 2018

@author: nicolas
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2
plt.rcParams['image.cmap'] = 'gray'

def alinear_region_UTE(img_color):
    #Input: region donde LED del contador de UTE
    #Output: region alineada a tamaño estándar 200x600
    
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    #Crear imagen binario con zona lED en negro
    ret,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((5,5),np.uint8)
    close = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel) 
    kernel2 = np.ones((5,5),np.uint8)
    img_bin = cv2.morphologyEx(close, cv2.MORPH_DILATE, kernel2) 
    
    #Hallo las esquinas
    ptos = np.argwhere(img_bin < 250)
    suma = np.sum(ptos, axis=1) 
    resta = np.diff(ptos, axis=1)

    A = ptos[np.argmin(suma)]
    B = ptos[np.argmin(resta)]
    C = ptos[np.argmax(suma)]
    D = ptos[np.argmax(resta)]
    
    #Hago la homografía 
    pts_src = np.array([A,B,C,D])
    pts_dst = np.array([[0,0],[200,0],[200,650],[0,650]])
    h, status = cv2.findHomography(pts_src, pts_dst)    
    img_alineada = cv2.warpPerspective(img, h, (650,200))
    
    return img_alineada
#%%
#Carga de archivos
regions = []
for i in [3, 4, 5, 6, 7, 8, 9, 10, 11]:
    filename = '../images/regiones_UTE/region_ute{}.jpg'.format(i)
    img = cv2.imread(filename)
    regions.append(img)
#%%
img_color = regions[0]
img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
plt.figure(); plt.imshow(img, cmap= 'gray')
#%%
#Aplico un thershold de otsu y realizo operaciones morfologicas

ret,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kernel = np.ones((5,5),np.uint8)
close = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel) 
kernel2 = np.ones((5,5),np.uint8)
img_bin = cv2.morphologyEx(close, cv2.MORPH_DILATE, kernel2) 
plt.figure(); plt.imshow(img_bin, cmap= 'gray')


#%%
#Hallo las esquinas
ptos = np.argwhere(img_bin < 200)
suma = np.sum(ptos, axis=1)
resta = np.diff(ptos, axis=1)

A = ptos[np.argmin(suma)]
B = ptos[np.argmin(resta)]
C = ptos[np.argmax(suma)]
D = ptos[np.argmax(resta)]

#Verificacion de los resultados
plt.plot(A[1], A[0], 'o')
plt.plot(B[1], B[0], 'o')
plt.plot(C[1], C[0], 'o')
plt.plot(D[1], D[0], 'o')
plt.imshow(img_bin)
#%%
pts_src = np.array([A,B,C,D])
pts_dst = np.array([[0,0],[200,0],[200,650],[0,650]])

h, status = cv2.findHomography(pts_src, pts_dst)    
img_alineada = cv2.warpPerspective(img, h, (650,200))
plt.imshow(img_alineada)
#%%

img_alineada = alinear_region_UTE(regions[8])
plt.close('all')
plt.figure(); plt.imshow(img_alineada)