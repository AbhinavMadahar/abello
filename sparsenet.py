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

import argparse
import sys
import os
import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random 


def longest_shortest_path(G):
    longest_path = []
    for src, paths in nx.all_pairs_shortest_path(G):
        for dest, path in paths.items():
            if len(path) > len(longest_path):
                longest_path = path
    return longest_path


def point_farthest_from_configuration(distance_matrix: np.array, configuration, vertex_to_index):
    distance_matrix = np.array(distance_matrix)
    points_in_configuration = set(vertex_to_index[vertex] for vertex in set(sum(configuration, [])))
    distance_matrix = distance_matrix[list(points_in_configuration)]
    distances_to_configuration = distance_matrix.min(axis=0)
    distances_to_configuration[distances_to_configuration == np.inf] == 0
    farthest_point = distances_to_configuration.argmax()
    index_to_vertex = { i:v for v, i in vertex_to_index.items() }
    return index_to_vertex[farthest_point], index_to_vertex[distance_matrix[:, farthest_point].argmin()]


if __name__ == '__main__':
    G = nx.Graph()
    G = nx.readwrite.gexf.read_gexf(sys.stdin).to_undirected()
    print('Loaded graph', file=sys.stderr)
    print('TOTAL NUMBER OF NODES:', G.number_of_nodes(), file=sys.stderr)
    print('TOTAL NUMBER OF EDGES:', G.number_of_edges(), file=sys.stderr)
    
    connected_components = sorted(list(nx.connected_components(G)), key=lambda comp: len(comp))[::-1]

    for i, connected_nodes in enumerate(connected_components):
        print('CONNECTED COMPONENT %s'%i, file=sys.stderr)

        connected_component = G.subgraph(connected_nodes)
        print('\tNUMBER OF NODES:', connected_component.number_of_nodes(), '(%s%%)'%(connected_component.number_of_nodes()/G.number_of_nodes()), file=sys.stderr)
        print('\tNUMBER OF EDGES:', connected_component.number_of_edges(), '(%s%%)'%(connected_component.number_of_edges()/G.number_of_nodes()), file=sys.stderr)
        vertex_name_to_index = { node:i for i, node in enumerate(connected_component.nodes) }

        start_time = time.time()
        try:
            distance_matrix = np.fromfile('distance_matrices/component-%s.npy'%i)
            print('\tLOADED DISTANCE MATRIX FROM FILE', file=sys.stderr)
        except:
            distance_matrix = np.array(nx.floyd_warshall_numpy(connected_component))
            np.save('distance_matrices/component-%s.npy'%i, distance_matrix)
            print('\tCALCULATED AND SAVED DISTANCE MATRIX', file=sys.stderr)
        print('\tDiameter =', distance_matrix.max(), file=sys.stderr)
        print('\tTook', time.time()-start_time, 'seconds to compute the distance matrix', file=sys.stderr)

        pos = nx.spring_layout(connected_component)
        nx.draw_networkx_nodes(connected_component, pos, node_color='black', node_size=1)
        nx.draw_networkx_edges(connected_component, pos, node_size=1)
        plt.savefig('plots/component-%s-0.png'%i)
        plt.clf()

        # initialise the configuration by adding the longest shortest path
        configuration = []

        # TODO: make this run more quickly
        # we know it takes `diameter` distance to get from u to v, so we might use a `diameter`-distance DFS
        configuration.append(longest_shortest_path(connected_component))  

        nx.draw_networkx_nodes(connected_component, pos, node_color='black', node_size=1)
        nx.draw_networkx_nodes(connected_component, pos, nodelist=sum(configuration, []), node_color='r', node_size=1)
        nx.draw_networkx_edges(connected_component, pos)
        plt.savefig('plots/component-%s-1.png'%i)
        plt.clf()

        # add the other paths until all points are either on the configuration or adjacent to it
        colors = ['red']
        while True:
            farthest_from_config, closest_point_on_config = point_farthest_from_configuration(distance_matrix, configuration, vertex_name_to_index)
            path = nx.shortest_path(connected_component, farthest_from_config, closest_point_on_config)
            if len(path) < 3:
                break
            configuration.append(path)
            print('\t%s'%len(configuration), 'paths in configuration', file=sys.stderr)
            nx.draw_networkx_nodes(connected_component, pos, node_color='black', node_size=1)
            nx.draw_networkx_edges(connected_component, pos)
            colors.append('#%02X%02X%02X'%(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
            for path, color in list(zip(configuration, colors))[::-1]:
                nx.draw_networkx_nodes(connected_component, pos, nodelist=path, node_color=color, node_size=1)
                edges = [(src, dest) for src, dest in zip(path, path[1:])]
                nx.draw_networkx_edges(connected_component, pos, edgelist=edges, edge_color=color)
            plt.savefig('plots/component-%s-%s.png'%(i, len(configuration)))
            plt.clf()

        print('\tDone generating configurations', file=sys.stderr)
        framerate = len(configuration) // 15 or 1  # we want the video to be around 15 seconds, and we have the `or 1` for if len(config) < 15
        os.system("ffmpeg -r {} -pattern_type glob -i 'plots/component-{}-*.png' plots/component-{}-graph.mp4 2>/dev/null".format(framerate, i, i))
