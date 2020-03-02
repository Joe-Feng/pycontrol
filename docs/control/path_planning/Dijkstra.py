import numpy as np
import cv2
from pycontrol.control import path_planning


shape = (100,100)
map = np.ones(shape=shape)
map[0:60, 10:13] = 0
map[30:85, 30:33] = 0
map[35:38, 13:30] = 0
map[20:100, 50:53] = 0
map[30:80, 70:73] = 0
map[55:58, 73:90] = 0


start = [0,0]
end = [60,80]
closelist, path = path_planning.DijkstraMap(map, start=start, end=end)


h,w = shape
GridMap = np.ones(shape=(h,w,3))*255
GridMap[map==0] = [0,0,0]

GridMap[closelist[:,0],closelist[:,1]] = [0,0,255]

GridMap[start[0],start[1]] = GridMap[end[0],end[1]] = [0,255,0]

for i in range(path.shape[0]):
    point1 = (path[i][1],path[i][0])
    if point1 == (end[1],end[0]):
        break
    point2 = (path[i+1][1],path[i+1][0])
    cv2.line(GridMap, point1, point1, color=(255,0,0))


cv2.namedWindow('Dijkstra', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Dijkstra',500,500)
cv2.imshow('Dijkstra', GridMap)
cv2.waitKey()


