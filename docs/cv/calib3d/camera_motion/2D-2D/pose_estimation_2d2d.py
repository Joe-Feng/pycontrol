import cv2
import numpy as np




# 读取图像
# read image
img1 = cv2.imread('../../../data/1.png')
img2 = cv2.imread('../../../data/2.png')
assert img1 is not None and img2 is not None



orb = cv2.ORB_create()

keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

