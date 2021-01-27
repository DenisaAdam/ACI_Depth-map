# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:30:42 2021

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

left = cv2.imread('im4.png')
right = cv2.imread('im5.png')

left = ResizeWithAspectRatio(left, width=800)
right = ResizeWithAspectRatio(right, width=800)

left = cv2.pyrDown(left)
right = cv2.pyrDown(right)

height, width, _ = left.shape
print(width)

leftGray = np.zeros((height,width,1), np.uint8)
rightGray = np.zeros((height,width,1),np.uint8)
a = 0.2989
b = 0.587
c = 0.072

for i in range(height):
    for j in range(width):
        leftGray[i,j]= a*left[i,j,2] + b*left[i,j,1] + c*left[i,j,0]
        rightGray[i,j]= a*right[i,j,2] + b*right[i,j,1] + c*right[i,j,0]

SAD = np.zeros((height, width,1),np.uint8)
disparity = np.zeros((height, width,1),np.uint8)
depth = np.zeros((height, width,1),np.uint8)


disparityRange = 40
blockSize = 12
halfBlockSize = int(blockSize/2)

offset_adjust = 255 / disparityRange*2
for y in range(halfBlockSize, height - halfBlockSize):
    for x in range(halfBlockSize, width - halfBlockSize):
        min = 1000;
        best_offset = 0
        for offset in range (-disparityRange, disparityRange):
            SAD[y,x]=0
            for i in range (-halfBlockSize, halfBlockSize):
                for j in range (-halfBlockSize, halfBlockSize):
                    u = y+i
                    v= x+j+offset
                    if(v < width and v >= 0):
                        SAD[y,x] += leftGray[u,x+j] - rightGray[u,v]
            if(SAD[y,x] <= min):
                min = abs(SAD[y,x])
                best_offset = offset
        SAD[y,x] = min
        for i in range(-halfBlockSize,halfBlockSize):
              for j in range(-halfBlockSize,halfBlockSize):
                  SAD[y+i,x+j] = min
                  if((x+best_offset) < width and (x+best_offset) >= 0):
                      #disparity[y+i,x+j]= best_offset * offset_adjust
                      disparity[y+i,x+j]= abs(leftGray[y,x] - rightGray[y,x + best_offset])
                           

baseline = 170
f = 461

for i in range(0, height):
    for j in range(0, width):
        if(disparity[i,j] > 0):
          depth[i,j] = int(f* baseline / disparity[i,j])

cv2.imshow("SAD", SAD)
cv2.imshow("disparity", disparity)
cv2.imshow("depth", depth)
cv2.waitKey()
cv2.destroyAllWindows()