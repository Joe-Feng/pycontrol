from pycontrol.data_structure import graph
import numpy as np



Graph = np.inf*np.ones(shape=(7,7))
Graph[0][0] = 0
Graph[0][1] = 2
Graph[0][2] = 4
Graph[0][3] = 1
Graph[1][1] = 0
Graph[1][3] = 3
Graph[1][4] = 10
Graph[2][2] = 0
Graph[2][3] = 2
Graph[2][5] = 5
Graph[3][3] = 0
Graph[3][4] = 2
Graph[3][5] = 8
Graph[3][6] = 4
Graph[4][4] = 0
Graph[4][6] = 6
Graph[5][5] = 0
Graph[5][6] = 1
Graph[6][6] = 0
Graph = np.triu(Graph)
Graph += Graph.T - np.diag(Graph.diagonal())


path = graph.Dijkstra(Graph, start=0, end=5)
print(path)
