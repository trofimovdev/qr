# -*- coding: utf-8 -*-
from cv2 import *
import numpy as np

cam = cv2.VideoCapture(0)
width, height = 480, 640

#~ cam.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
#~ cam.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)
#~ cam.set(cv2.cv.CV_CAP_PROP_FPS, 25)


while True:
	ret_val, img = cam.read()
	r, g, b = split(img)
	
	region = [[b[width/10 * w:width/10 * (w+1), height/10 * h:height/10 * (h+1)].sum(axis = None) for h in range(10)] for w in range(10)]
	#~ print(region)
	#~ a = np.where(b < 50)
	#~ if a[0].shape[0] > 0:
		#~ print('Синяя точка!')
		#~ for i in range(len(a[1]) - 3, len(a[1])):
			#~ circle(b, (a[1][i], a[0][i]), 10, 255, 3)
	
	lines = HoughLines(b, 1000, np.pi / 180, 200)
	for rho,theta in lines[0]:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))

		line(b, (x1, y1), (x2, y2), (0, 0, 255), 2)
	#~ for w in range(10):
		#~ for h in range(10):
			#~ if region[w][h] < 350000:
				#~ circle(b, (height/10 * h + 32, width/10 * w + 24), 10, 255, 3)
	imshow('window', b)
	waitKey(10)
