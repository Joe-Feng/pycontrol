import cv2
import numpy as np
from pycontrol import pcv, params


# 读取图像
# read image
img1 = cv2.imread('../../../data/1.png')
img2 = cv2.imread('../../../data/2.png')
assert img1 is not None and img2 is not None


def getMatches(img1, img2):
    orb = cv2.ORB_create()

    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(descriptors1, descriptors2)

    matches = sorted(matches, key=lambda x: x.distance)
    min_dist = matches[0].distance

    goodmatches = []
    for i in range(len(descriptors1)):
        if matches[i].distance <= max(2 * min_dist, 30):
            goodmatches.append(matches[i])

    return keypoints1, keypoints2, goodmatches



keypoints1, keypoints2, matches = getMatches(img1, img2)
points1 = []
points2 = []
for i in range(len(matches)):
    points1.append(keypoints1[matches[i].queryIdx].pt)
    points2.append(keypoints2[matches[i].trainIdx].pt)



# 计算本质矩阵
# calculate Essential Matrix
fx = fy = 521
cx = 325.1
cy = 249.7
K = [fx, fy, cx, cy]
essential_matrix = pcv.findEssentialMat(points1, points2, K, method=params.FM_8POINT)
print('essential matrix is:\n', essential_matrix)




