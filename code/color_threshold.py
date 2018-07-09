#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 17:08:42 2018

@author: franchesoni
"""
import numpy as np

def color_threshold(img, color=[99, 87, 77], e=[15, 25, 35]):
    l, u = np.empty_like(img), np.empty_like(img)
    for i in range(3):
        l[:,:,i] = color[i] - e[i]
        u[:,:,i] = color[i] + e[i]
    thresh = np.prod((img > l) * (img < u) * 1, axis=2)
    thresh = np.expand_dims(thresh, 2)
    return thresh
