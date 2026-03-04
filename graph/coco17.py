import sys

import numpy as np
sys.path.extend(['../'])
from graph import tools
def edge2mat(edges, num_node):
    A = np.zeros((num_node, num_node), dtype=np.float32)
    for i, j in edges:
        A[j, i] = 1
    return A

def normalize_digraph(A):
    Dl = np.sum(A, axis=0)
    Dn = np.zeros((A.shape[0], A.shape[0]), dtype=np.float32)
    for i in range(A.shape[0]):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    return A @ Dn

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = 17
        self.self_link = [(i, i) for i in range(self.num_node)]

        # COCO-17 undirected edges
        neighbor_undirected = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6),
            (5, 7), (7, 9),
            (6, 8), (8, 10),
            (5, 11), (6, 12),
            (11, 12),
            (11, 13), (13, 15),
            (12, 14), (14, 16),
        ]

        inward = neighbor_undirected
        outward = [(j, i) for (i, j) in inward]
        self.outward = outward
        self.inward = inward
        self.neighbor = inward + outward
        I = edge2mat(self.self_link, self.num_node)
        In = normalize_digraph(edge2mat(inward, self.num_node))
        Out = normalize_digraph(edge2mat(outward, self.num_node))
        self.A = np.stack((I, In, Out)).astype(np.float32)
        self.A_outward_binary = tools.get_adjacency_matrix(self.outward, self.num_node)