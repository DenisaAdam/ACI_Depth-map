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

height, width, _ = left.shape
#print(height)
#print(width)
#print(left.shape)
#print(left.size)

pair = np.zeros((height, width))
print(pair)

for x in range(height-1):
    for y in range(width-1):
        (b, g, r) = left[x,y]
        for i in [-5,5]:
            (b0, g0, r0) = right[x,y]
            s0=1000;
            if y + i >= 0 and y + i <= width-1:
                (b1, g1, r1)=right[x, y + i]
                s = (b-b1)+(g-g1)+(r-r1)
                if s < s0:
                    pair[x][y] = s
                
print(pair)

#cv2.imshow('original left', left)
#cv2.imshow('original right', right)
cv2.waitKey()
cv2.destroyAllWindows()