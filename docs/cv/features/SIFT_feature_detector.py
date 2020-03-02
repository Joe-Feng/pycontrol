import cv2
from pycontrol import pcv
import time




image_src = cv2.imread('../data/grass.jpg')
image = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)


start = time.time()
keyPoints = pcv.siftDetectAndCompute(image)
print(time.time() - start)


h, w = image.shape
for n in range(len(keyPoints)):
    j,i = keyPoints[n]
    cv2.circle(image_src, (int(round(i)),int(round(j))), radius=2, color=(0,0,255), thickness=2)


cv2.imshow('sift', image_src)
cv2.waitKey()


