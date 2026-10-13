import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from cleaner import clean_image

img1 = cv.imread('img/sample3.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('img/sample2.jpg', cv.IMREAD_GRAYSCALE)
 
# Initiate ORB detector
orb = cv.ORB.create(nfeatures=2000, fastThreshold=10)

brisk = cv.BRISK_create()

img1 = clean_image(img1)
 
# find the keypoints with ORB
kp = orb.detect(img1,None)
 
# compute the descriptors with ORB
kp, des = orb.compute(img1, kp)
 
# draw only keypoints location,not size and orientation
# drawn = cv.drawKeypoints(img1, kp, None, color=(0,255,0), flags=0)
# plt.imshow(drawn), plt.show()
