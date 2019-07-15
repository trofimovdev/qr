# -*- coding: utf-8 -*-
import numpy as np
import cv2
from math import *
from time import *
from numba import *

print('NumPy', np.__version__)
print('OpenCV', cv2.__version__)


img = cv2.imread('qr.jpg')
img_canvas = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(11)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 121, 1)
print(22)
# edges = cv2.Canny(thresh, 100, 200)
print(33)
@jit(parallel = True)
def findC():
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    print(hierarchy)
    cnt = []
    for i, contour in enumerate(contours):
        if hierarchy[0][i][1] == -1:
            cnt.append(contour)
    return cnt
cnt = findC()
c = sorted(cnt, key = cv2.contourArea, reverse = True)
for i in range(3):
    x,y,w,h = cv2.boundingRect(c[i])
    cv2.rectangle(img_canvas,(x,y),(x+w,y+h),(0,255,0), 3)
    cv2.putText(img_canvas, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255))




h, w = img.shape[:2]
s = 1
while h // s > 600: s += 1
def imshow(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, img.shape[1] // s, img.shape[0] // s)
    return cv2.imshow(name, img)

imshow('img_canvas', img_canvas)
imshow('thresh', thresh)

cv2.waitKey(0)
