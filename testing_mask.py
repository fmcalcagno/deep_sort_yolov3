#testing masks

import cv2
import numpy as np

test= cv2.imread("test.jpg")

mask= cv2.imread("mask.png",0)

im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP , cv2.CHAIN_APPROX_TC89_KCOS)
mask= cv2.imread("mask.png")
cv2.drawContours(mask, contours, -1, (255,255,0),3)

cv2.imshow('',mask)
cv2.waitKey()
