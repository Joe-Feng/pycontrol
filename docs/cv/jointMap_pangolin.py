import sys
from pycontrol import conf
sys.path.append(conf.pangolin_build_src)

import numpy as np
import cv2
import numba as nb
from pycontrol import pcv, mat
import pypangolin as pango
from OpenGL.GL import *
import time


filename = './data/pose.txt'

fx = 518.0
fy = 519.0
cx = 325.5
cy = 253.5
K = np.array([fx,fy,cx,cy])
depthScale = 1000.0



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
            glColor3d(p[3]/255.0, p[4]/255.0, p[5]/255.0)
            glVertex3d(p[0], p[1], p[2])
        glEnd()

        pango.FinishFrame()
        time.sleep(0.005)




def read_file(filename):
    poses = []
    with open(filename, 'r') as f:
        for l in f.readlines():
            l = l.split()
            tx, ty, tz, qx, qy, qz, qw = map(float, l)
            q = np.array([qw, qx, qy, qz])
            t = np.array([tx, ty, tz])
            SE3 = mat.SE3_qt(q, t)
            poses.append(SE3)

    return np.array(poses)


if __name__ == '__main__':
    colorImgs = nb.typed.List()
    depthImgs = nb.typed.List()
    for i in range(5):
        color_img = cv2.imread('./data/color/'+str(i+1)+'.png')
        colorImgs.append(color_img)
        depth_img = cv2.imread('./data/depth/'+str(i+1)+'.pgm', cv2.IMREAD_UNCHANGED)
        depthImgs.append(depth_img)

    poses = read_file(filename)

    start = time.time()
    pointcloud = pcv.joint_map(colorImgs, depthImgs, poses, K, depthScale)
    print(time.time() - start)

    print('a total of ' + str(len(pointcloud)) + ' points.')
    showPointCloud(pointcloud)


