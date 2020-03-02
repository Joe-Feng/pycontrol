import cv2
import numpy as np
from pycontrol import pcv
import time



image_src = cv2.imread('../data/grass.jpg')
image = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
image = cv2.medianBlur(image, 7)

start = time.time()
keyPoints = pcv.orbDetectAndCompute(image)
print(time.time() - start)


h, w = image.shape
for j in range(h):
    for i in range(w):
        if keyPoints[j,i] > 0:
            cv2.circle(image_src, (int(round(i)), int(round(j))), radius=2, color=(0, 0, 255), thickness=2)

cv2.namedWindow('orb', cv2.WINDOW_NORMAL)
cv2.imshow('orb', image_src)
cv2.waitKey()


