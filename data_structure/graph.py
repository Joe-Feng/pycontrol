import numpy as np





def Dijkstra(graph, start, end):
    numNode = graph.shape[0]
    closelist = [start]
    closelist = np.array(closelist)
    openlist = np.array([i for i in np.arange(numNode, dtype=np.int32)])
    distlist = np.array([np.inf for _ in np.arange(numNode)])
    nodelist = np.array([-1 for _ in np.arange(numNode, dtype=np.int32)])


    openlist = np.delete(openlist, start)

    # 相邻的点
    nearNode = (graph[start] != np.inf)
    nearNode[closelist] = False

    index = np.array([i for i in np.arange(numNode, dtype=np.int32)])
    index = index[nearNode]

    # 更新distlist
    distlist[start] = 0
    distlist[index] = graph[start][index]

    # 更新nodelist
    nodelist[start] = start
    nodelist[index] = start

    while 1:
        # 距初始点最小距离的点
        argmin = np.argmin(distlist[index])
        minDistNode = index[argmin]

        # 将最小距离的点加入closelist
        closelist = np.append(closelist, minDistNode)
        openlist = np.delete(openlist, np.where(openlist==minDistNode)[0])
        if minDistNode == end:
            break

        # 和最小距离的点相邻的点
        nearNode = (graph[minDistNode] != np.inf)
        nearNode[closelist] = False

        index = np.array([i for i in np.arange(numNode, dtype=np.int32)])
        index = index[nearNode]


        # 和最小距离的点相邻的点到初始点的最短距离
        dy = np.min(
            [distlist[index],
             distlist[minDistNode]+graph[minDistNode][index]],
             axis=0
        )

        isUpdate = np.argmin(
            [distlist[index],
             distlist[minDistNode] + graph[minDistNode][index]],
            axis=0
        )

        # 更新distlist
        distlist[index] = dy

        # 更新nodelist
        nodelist[index[np.where(isUpdate==1)[0]]] = minDistNode



    path = [end]
    index = nodelist[end]
    path.append(index)
    while 1:
        value = nodelist[index]
        if index == value:
            break

        path.append(value)
        index = value

    return np.array(list(reversed(path)))


