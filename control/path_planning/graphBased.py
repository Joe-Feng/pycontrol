import numpy as np
from pycontrol.cv import iproc




def DijkstraMap(map, start, end):
    h, w = map.shape
    closelist = [start]

    distlist = np.inf * np.ones(shape=(h,w))
    nodelist = -1 * np.ones(shape=(h,w,2), dtype=np.int64)

    # 相邻的点
    nearNode = []
    neighbors = iproc.neighbor4(start).tolist()
    for i in range(len(neighbors)):
        if neighbors[i][0]<0 or neighbors[i][1]<0 \
           or neighbors[i][0]>=h or neighbors[i][1]>=w:
            continue
        elif neighbors[i] in closelist:
            continue
        elif map[neighbors[i][0], neighbors[i][1]] == 0:
            continue

        nearNode.append(neighbors[i])
    nearNode = np.array(nearNode)

    # 更新distlist
    distlist[start[0],start[1]] = 0
    distlist[nearNode[:,0], nearNode[:,1]] = 1

    # 更新nodelist
    nodelist[start[0],start[1]] = start
    nodelist[nearNode[:,0], nearNode[:,1]] = start

    # print(nearNode)
    while 1:
        # 距初始点最小距离的点
        argmin = np.where(np.min(distlist[nearNode[:,0], nearNode[:,1]])
                          ==distlist[nearNode[:,0], nearNode[:,1]])

        minDistNode = nearNode[argmin].tolist()

        # 将最小距离的点加入closelist
        closelist.extend(minDistNode)
        if end in minDistNode:
            break

        # 和最小距离的点相邻的点
        nearNode = []
        nearNodeSet = []
        for j, node in enumerate(minDistNode):
            neighbor = []
            neighbors = iproc.neighbor4(node).tolist()
            for i in range(len(neighbors)):
                if neighbors[i][0] < 0 or neighbors[i][1] < 0 \
                        or neighbors[i][0] >= h or neighbors[i][1] >= w:
                    continue
                elif neighbors[i] in closelist:
                    continue
                elif map[neighbors[i][0], neighbors[i][1]] == 0:
                    continue

                neighbor.append(neighbors[i])
                nearNodeSet.append(tuple(neighbors[i]))

            if len(neighbor) == 0:
                minDistNode = np.delete(minDistNode,j,axis=0)
                continue
            nearNode.append(neighbor)


        # 和最小距离的点相邻的点到初始点的最短距离
        for i in range(len(nearNode)):
            near = np.array(nearNode[i])
            # print(near)
            dy = np.min(
                [distlist[near[:,0], near[:,1]],
                 distlist[minDistNode[i][0], minDistNode[i][1]]+np.ones(shape=near.shape[0])],
                 axis=0
            )

            isUpdate = np.argmin(
                [distlist[near[:, 0], near[:, 1]],
                 distlist[minDistNode[i][0], minDistNode[i][1]] + np.ones(shape=near.shape[0])],
                axis=0
            )

            # 更新distlist
            distlist[near[:,0], near[:,1]] = dy

            # 更新nodelist
            nodelist[near[np.where(isUpdate==1)[0]][:,0],
                     near[np.where(isUpdate==1)[0]][:,1]] = minDistNode[i]

        nearNode = np.array(list(set(nearNodeSet)))

    path = [end]
    index = nodelist[end[0],end[1]]
    path.append(index)
    while 1:
        value = nodelist[index[0],index[1]]
        if index.tolist() == value.tolist():
            break

        path.append(value)
        index = value

    return np.array(closelist), np.array(list(reversed(path)))



