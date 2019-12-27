import sys
from pycontrol import conf
sys.path.append(conf.pangolin_build_src)

import pypangolin as pango
from pycontrol import mat
from OpenGL.GL import *
import numpy as np
import time


groundtruth_file = './data/groundtruth.txt'
estimated_file = './data/estimated.txt'
poses = []


def DrawTrajectory(groundtruth, estimated):
    pango.CreateWindowAndBind("trajectory viewer", 1024, 768)
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
                    -1024/768)
    d_cam.SetHandler(handler)

    while not pango.ShouldQuit():
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        d_cam.Activate(s_cam)
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glLineWidth(2)

        for i in range(len(groundtruth)-1):
            glColor3f(0.0, 1.0, 0.0)
            glBegin(GL_LINES)

            SE3_p1 = groundtruth[i]
            SE3_p2 = groundtruth[i+1]
            glVertex3d(SE3_p1[0], SE3_p1[1], SE3_p1[2])
            glVertex3d(SE3_p2[0], SE3_p2[1], SE3_p2[2])
            glEnd()

        for i in range(len(estimated)-1):
            glColor3f(1.0, 0.0, 0.0)
            glBegin(GL_LINES)

            SE3_p1 = estimated[i]
            SE3_p2 = estimated[i + 1]
            glVertex3d(SE3_p1[0], SE3_p1[1], SE3_p1[2])
            glVertex3d(SE3_p2[0], SE3_p2[1], SE3_p2[2])
            glEnd()

        pango.FinishFrame()
        time.sleep(0.005)



def read_file(filename):
    trajectory = []
    with open(filename, 'r') as f:
        for l in f.readlines():
            l = l.split()
            _time = l[0]
            tx, ty, tz, qx, qy, qz, qw = map(float, l[1:])
            q = np.array([qw, qx, qy, qz])
            t = np.array([tx, ty, tz])
            SE3 = mat.SE3_qt(q, t)
            trajectory.append(SE3)

    return np.array(trajectory)



if __name__ == '__main__':
    groundtruth = read_file(groundtruth_file)
    estimated = read_file(estimated_file)
    assert(len(groundtruth)!=0 and len(estimated)!=0)
    assert(len(groundtruth) == len(estimated))

    rmse = 0
    for i in range(len(estimated)):
        SE3_p1 = estimated[i]
        SE3_p2 = groundtruth[i]

        p2_inv = mat.SE3_inverse(SE3_p2)
        temp1 = mat.SE3_mul_SE3(p2_inv, SE3_p1)
        temp2 = mat.SE3_log(temp1)
        error = np.sqrt(np.sum(np.power(temp2, 2)))
        rmse += error ** 2

    rmse = rmse / len(estimated)
    rmse = np.sqrt(rmse)

    print("RMSE = ", rmse)

    DrawTrajectory(groundtruth, estimated)
