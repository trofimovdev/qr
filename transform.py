# -*- coding: utf-8 -*-
import numpy as np
import cv2
from math import *
import re
#from numba import *
from time import *
print(np.__version__)
img = cv2.imread('realdistortions.png', 0)
thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
#~ canny = cv2.Canny(thresh, 0, 0, apertureSize = 3, L2gradient = True)
h, w = img.shape
s = 1
while h // s > 600: s += 1
def imshow(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, img.shape[1] // s, img.shape[0] // s)
    return cv2.imshow(name, img)





img_canvas = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)
thresh_canvas = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)


vect = np.diff(thresh)
ratio_f = np.frompyfunc(lambda row: np.diff(np.argwhere(vect[row] != 0).flatten() + 1, prepend = 0, append = vect.shape[0] + 2), 1, 1)
ratio_x = ratio_f(np.arange(thresh.shape[0]))
vect = np.diff(thresh.T)
ratio_y = ratio_f(np.arange(thresh.shape[1]))

centers = []

for i in range(len(ratio_x)):
    for e in range(0, len(ratio_x[i]) - 5):
        if np.array_equal((np.array(ratio_x[i][e : e + 5], dtype = np.single) / ratio_x[i][e]).round(), np.array([1, 1, 3, 1, 1])):
            centers += [((sum(ratio_x[i][:e]) + sum(ratio_x[i][:e + 5])) // 2, i)]

for i in range(len(ratio_y)):
    for e in range(0, len(ratio_y[i]) - 5):
        if np.array_equal((np.array(ratio_y[i][e: e + 5],dtype = np.single) / ratio_y[i][e]).round(), np.array([1, 1, 3, 1, 1])):
            centers += [(i, (sum(ratio_y[i][:e]) + sum(ratio_y[i][:e + 5])) // 2)]



centers = np.array(list(set(filter(lambda i: centers.count(i) > 1, centers))))


print(centers)
for center in centers:
    cv2.circle(img_canvas, (center[0], center[1]), 2, (0, 0, 255), 2)
distance = {}
for first in range(centers.shape[0] - 1):
	for second in range(first + 1, centers.shape[0]):
		distance.update({np.linalg.norm(centers[first] - centers[second]): (centers[first], centers[second])})

print(distance)

print(distance[max(distance)])


d = distance[max(distance)]
theta = 45 - np.arctan(1.0 * np.abs(d[0][1] - d[1][1]) / np.abs(d[0][0] - d[1][0])) * 180 / np.pi
print(theta)
#~ cRow, cCol = np.abs(d[0][0] - d[1][0]) // 2, np.abs(d[0][1] - d[1][1]) // 2
cRow, cCol = img_canvas.shape[0] // 2, img_canvas.shape[1] // 2



M = cv2.getRotationMatrix2D((cRow, cCol), theta, 1)
img_canvas = cv2.warpAffine(img_canvas, M, (img_canvas.shape[0], img_canvas.shape[1]))





imshow('img_canvas', img_canvas)
imshow('thresh_canvas', thresh_canvas)





imshow('img', img)
imshow('tresh', thresh)


cv2.waitKey(0)
