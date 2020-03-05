import sys
from pycontrol import conf
sys.path.append(conf.pangolin_build_src)

import cv2
import numpy as np
from pycontrol import pcv
import pypangolin as pango
from OpenGL.GL import *
import time



fx = 718.856
fy = 718.856
cx = 607.1928
cy = 185.2157
K = np.array([fx, fy, cx, cy])
b = 0.573


left_file = '../data/left.png'
right_file = '../data/right.png'


def showPointCloud(pointcloud):
    if len(pointcloud) == 0:
        raise Exception("point cloud could not be empty!")

    pango.CreateWindowAndBind("pointcloud viewer", 1024, 768)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    pm = pango.ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000)
    mv = pango.ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    s_cam = pango.OpenGlRenderState(pm, mv)

    handler = pango.Handler3D(s_cam)
    d_cam = pango.CreateDisplay()
    d_cam.SetBounds(pango.Attach(0.0), pango.Attach(1.0),
                    pango.Attach(0.0), pango.Attach(1.0),
                    -1024 / 768)
    d_cam.SetHandler(handler)

    while not pango.ShouldQuit():
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        d_cam.Activate(s_cam)
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glPointSize(2)
        glBegin(GL_POINTS)

        for p in pointcloud:
            glColor3d(p[3], p[3], p[3])
            glVertex3d(p[0], p[1], p[2])
        glEnd()

        pango.FinishFrame()
        time.sleep(0.005)


if __name__ == '__main__':
    left_image = cv2.imread(left_file, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_file, cv2.IMREAD_GRAYSCALE)

    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=96, blockSize=9,
        P1=8*9*9, P2=32*9*9, disp12MaxDiff=1, preFilterCap=63,
        uniquenessRatio=10, speckleWindowSize=100, speckleRange=32)

    disparity = sgbm.compute(left_image, right_image).astype(np.float32) / 16.0

    start = time.time()
    pointcloud = pcv.computeDepth(disparity, left_image, K, b)
    print(time.time() - start)

    showPointCloud(pointcloud)
    cv2.imshow('disparity', disparity / 96.0)
    cv2.waitKey()


