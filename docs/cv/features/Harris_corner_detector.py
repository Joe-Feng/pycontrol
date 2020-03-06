import cv2
from pycontrol import pcv, data, params
import numpy as np
import time



image_src = cv2.imread('../data/grid.jpg')
image = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)

start_t = time.time()
dst = pcv.cornerHarris(image, 3, 0.05)
dst = data.normalize(dst, 0, 255, normType=params.norm_minmax)
dst = pcv.cornerNMS(dst, (7,7))
print(time.time() - start_t)


h, w = dst.shape
for j in range(h):
    for i in range(w):
        if dst[j, i] > 120:
            cv2.circle(image_src, (i,j), radius=5, color=(0,0,255), thickness=2)

cv2.namedWindow('harris', cv2.WINDOW_NORMAL)
cv2.imshow('harris', image_src)
cv2.waitKey()




