import cv2
from pycontrol import pcv, data, params
import numpy as np



image = cv2.imread('../data/window.jpg', cv2.IMREAD_GRAYSCALE)


dst = pcv.cornerHarris(image, 2, 3, 0.05)
dst = data.normalize(dst, params.norm_minmax, params.image) * 255
dst = (dst*255).astype(np.uint8)
h,w = dst.shape
for j in range(h):
    for i in range(w):
        if dst[j, i] > 240:
            cv2.circle(image, (j,i), radius=2, color=(0,0,255))

cv2.imshow('img', image)
cv2.waitKey()