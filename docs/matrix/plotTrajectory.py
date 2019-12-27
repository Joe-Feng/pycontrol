import sys
from pycontrol import conf
sys.path.append(conf.pangolin_build_src)

import pypangolin as pango
from pycontrol import mat
from OpenGL.GL import *
import time


trajectory_file = './data/trajectory.txt'
poses = []


def DrawTrajectory(poses):
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

    poses.append(poses[0])
    while not pango.ShouldQuit():
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        d_cam.Activate(s_cam)
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glLineWidth(2)

        for i, pose in enumerate(poses[:-1]):
            _, t = mat.T2Rt(pose)
            Ow = t
            Xw = mat.transform_homogeneous(pose, list(map(lambda x:x*0.1,[1,0,0])))
            Yw = mat.transform_homogeneous(pose, list(map(lambda x:x*0.1,[0,1,0])))
            Zw = mat.transform_homogeneous(pose, list(map(lambda x:x*0.1,[0,0,1])))

            glBegin(GL_LINES)
            glColor3f(1.0, 0.0, 0.0)
            glVertex3d(Ow[0], Ow[1], Ow[2])
            glVertex3d(Xw[0], Xw[1], Xw[2])
            glColor3f(0.0, 1.0, 0.0)
            glVertex3d(Ow[0], Ow[1], Ow[2])
            glVertex3d(Yw[0], Yw[1], Yw[2])
            glColor3f(0.0, 0.0, 1.0)
            glVertex3d(Ow[0], Ow[1], Ow[2])
            glVertex3d(Zw[0], Zw[1], Zw[2])

            p1 = poses[i]
            p2 = poses[i+1]
            _, t1 = mat.T2Rt(p1)
            _, t2 = mat.T2Rt(p2)
            glColor3f(0.0, 0.0, 0.0)
            glVertex3d(t1[0], t1[1], t1[2])
            glVertex3d(t2[0], t2[1], t2[2])
            glEnd()

        pango.FinishFrame()
        time.sleep(0.005)



if __name__ == '__main__':
    with open(trajectory_file, 'r') as f:
        for l in f.readlines():
            l = l.split()
            _time = l[0]
            tx, ty, tz, qx, qy, qz, qw = map(float, l[1:])
            R = mat.quaternion2R([qw, qx, qy, qz])
            T = mat.Rt2T(R, [tx,ty,tz])
            poses.append(T)
    print('read total %d pose entries' % len(poses))

    DrawTrajectory(poses)


