import pcl
import pcl.pcl_visualization
import numpy as np
import cv2
import numba as nb
from pycontrol import pcv, mat
import time
import os


filename = './data/pose.txt'

fx = 518.0
fy = 519.0
cx = 325.5
cy = 253.5
K = np.array([fx,fy,cx,cy])
depthScale = 1000.0



#TODO 显示颜色
def showPointCloud(pointcloud):
    pointcloud = np.array(pointcloud)
    pointcloud = pointcloud[:,:3].astype(np.float32)
    # cloud = pcl.PointCloud_PointXYZRGB(pointcloud)
    cloud = pcl.PointCloud(pointcloud)
    pcl.save_XYZRGBA(cloud, './data/map.pcd', format='pcd')

    os.system('pcl_viewer_release ./data/map.pcd')


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


