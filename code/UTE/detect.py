#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 13:34:43 2018

@author: franchesoni
"""

import cv2

def detect(c):
    # initialize the shape name and approximate the contour
    shape = None
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.08 * peri, True)
    if (len(approx) == 4):
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = None if ar >= 0.95 and ar <= 1.05 else "rectangle"
    return shape

