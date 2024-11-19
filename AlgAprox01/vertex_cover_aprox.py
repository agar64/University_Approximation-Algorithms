import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import time

from networkx.algorithms.approximation import min_weighted_vertex_cover


G = nx.fast_gnp_random_graph(5, 0.25, seed=int(time.time()), directed=False)

def remove_edges(E):

    mask_A = np.isin(E, E[0])
    mask = np.logical_or.reduce(mask_A, 1)

    return E[~mask]


def approx_vertex_cover(G):
    C = np.empty((0, 2), dtype=int)
    E = np.array(G.edges)

    while(E.size):
        C = np.append(C, [E[0][0], E[0][1]])
        #C = np.append(C, [E[0].tolist()], axis=0) # <- Appends the edge instead
        E = remove_edges(E)

    return C

result = approx_vertex_cover(G).tolist()
result.sort()

print("Nodes:", np.array(G.nodes()),"\nEdges:\n",np.array(G.edges).tolist(), "\nVertex Cover:",
      result,"\nVertex Cover (Library):", min_weighted_vertex_cover(G))

nx.draw(G, with_labels=True)
plt.show()



#G = nx.complete_graph(100)
#
#print(G.edges)
#
# nx.draw(G)
# plt.show()

# E_linha = E
# for i in E[:, 0].size:
#     for j in E[0, :].size:
#         if E_linha[i][j] in {E[0][0], E[0][1]}: