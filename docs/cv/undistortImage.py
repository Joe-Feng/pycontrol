import cv2
import numpy as np
from pycontrol import pcv
import time



image_file = './data/distorted.png'

fx = 458.654
fy = 457.296
cx = 367.215
cy = 248.375
K = np.array([fx, fy, cx, cy])

k1 = -0.28340811
k2 = 0.07395907
p1 = 0.00019359
p2 = 1.76187114e-5
distort = np.array([k1, k2, p1, p2])


if __name__ == "__main__":
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    image_undistort = pcv.undistort(image, K, distort)

    cv2.imshow('img', image)
    cv2.imshow('img_undistort', image_undistort)
    cv2.waitKey()
