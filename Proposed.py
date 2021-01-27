# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:11:35 2020

@author: Denisa
"""

import cv2
import numpy as np

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

left = cv2.imread('im0.png')
right = cv2.imread('im1.png')

left = ResizeWithAspectRatio(left, width=800)
right = ResizeWithAspectRatio(right, width=800)

left = cv2.pyrDown(left)
right = cv2.pyrDown(right)


height, width, _ = left.shape

leftGray = np.zeros((height,width,1), np.uint8)
rightGray = np.zeros((height,width,1),np.uint8)
a = 0.2989
b = 0.587
c = 0.072

for i in range(height-1):
    for j in range(width-1):
        leftGray[i,j]= a*left[i,j,2] + b*left[i,j,1] + c*left[i,j,0]
        rightGray[i,j]= a*right[i,j,2] + b*right[i,j,1] + c*right[i,j,0]

SAO = np.zeros((height, width,1),np.uint8)
disparity = np.zeros((height, width,1),np.uint8)
depth = np.zeros((height, width,1),np.uint8)

disparityRange = 25
halfBlockSize = 7
blockSize = 2 * halfBlockSize + 1

offset_adjust = 255 / disparityRange*2  # this is used to map depth map output to 0-255 range

for i in range(0, height):
    for j in range(0, width):
            offset = 0
            min = 1000
            for y in range (-disparityRange, disparityRange+1):
                sao = 0
                for x in range(j - halfBlockSize, j + halfBlockSize+1):
                    if(x+y >= 0 and x+y <= width -1 and x >= 0 and x <= width -1):
                       SAO[i,j] += leftGray[i,x] - rightGray[i,x + y]
                if(SAO[i,j] < min):
                     min = SAO[i,j]
                     offset = y

            SAO[i,j] = abs(min) 
            disparity[i,j] = abs(offset * offset_adjust)
                        
baseline = 170
f = 461

for i in range(0, height):
    for j in range(0, width):
        if(disparity[i,j] != 0):
            depth[i,j] = int(f* baseline / disparity[i,j])

#cv2.imshow('original left', left)
#cv2.imshow('original right', right)
cv2.imshow("disparity", disparity)
cv2.imshow("depth", depth)
cv2.waitKey()
cv2.destroyAllWindows()