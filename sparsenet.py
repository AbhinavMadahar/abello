"""
SparseNet

Reads a graph file and generates a SparseNet representation for it.
It then outputs the SparseNet for it to STDOUT.

The graph file should start with a line that says how many vertices there are, and the remaining lines should be edges.
Note that the first vertex is the 0 vertex and the last one is the (n-1)th vertex.
For example,
    10
    0 7
    1 2
    4 2
    7 3
    1 9
"""

import networkx as nx
import numpy as np

def point_farthest_from_configuration(distance_matrix: np.array, configuration: list, vertex_to_index: dict) -> tuple:
    """
    Finds the point which is farthest away from any point currently on the configuration.
    """
    points_in_configuration = [vertex_to_index[vertex] for vertex in set(sum(configuration, []))]
    distance_matrix = distance_matrix[points_in_configuration]
    distances_to_configuration = distance_matrix.min(axis=0)
    farthest_point = distances_to_configuration.argmax()
    index_to_vertex = { i:v for v, i in vertex_to_index.items() }
    closest_on_configuration = points_in_configuration[distance_matrix[:, farthest_point].argmin()]
    return index_to_vertex[farthest_point], index_to_vertex[closest_on_configuration]


def sparsenet(G: nx.Graph, distance_matrix: np.array, vertex_to_index: dict):
    """
    Finds the SparseNet for a given graph.
    
    Args:
      G: a networkx graph
      distance_matrix: a 2d numpy array where distance_matrix[i][j] is the distance from node i to node j
      vertex_to_index: a dictionary which maps the names of vertices to an integer in {0, 1, 2, ..., |V|}.
    
    Returns a generator where each output is a path in the configuration.
    The first path is the longest path, 
    """
    index_to_vertex = { i:v for v, i in vertex_to_index.items() }
    src_index, dest_index = np.unravel_index(distance_matrix.argmax(), distance_matrix.shape)
    configuration = [nx.shortest_path(G, index_to_vertex[src_index], index_to_vertex[dest_index])]
    yield configuration[-1]
    while True:
        farthest_from_config, closest_point_on_config = point_farthest_from_configuration(distance_matrix, configuration, vertex_to_index)
        path = nx.shortest_path(G, farthest_from_config, closest_point_on_config)
        if len(path) < 3:
            break
        configuration.append(path)
        yield path

    
# we can run this file to find the SparseNet for a graph.
# to give the program the graph, we pass the edgelist file using STDIN.
# the program then outputs a list of lists representing the SparseNet.
if __name__ == '__main__':
    G = nx.Graph()
    with open('fabula/combined.csv', 'r') as combined:
        next(combined)  # ignore the header
        for edge in combined:
            src, dest = edge[:-1].split(',')
            G.add_edge(src, dest)
            G.add_node(src)
            G.add_node(dest)

    connected_component = G.subgraph(sorted(list(nx.connected_components(G)), key=lambda comp: len(comp))[-1])
    distance_matrix = np.load('distance_matrix.npy')
    vertex_name_to_index = { node:i for i, node in enumerate(connected_component.nodes) }
    
    for path in sparsenet(G, distance_matrix, vertex_name_to_index):
        print(path)