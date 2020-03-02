from pycontrol import pcv, params
import cv2
import time


image = cv2.imread('../data/window.jpg')

start = time.time()
# 均值滤波
# Mean filtering
dst1 = pcv.blur(image, shape=image.shape, ksize=3)
print('mean blur time: ', time.time() - start)

start = time.time()
# 高斯滤波
# Gaussian filtering
dst2 = pcv.GaussianBlur(image, shape=image.shape, ksize=3)
print('gaussian blur time: ', time.time() - start)

cv2.imshow('src', image)
cv2.imshow('mean blur', dst1)
cv2.imshow('gaussian blur', dst2)
cv2.waitKey()

