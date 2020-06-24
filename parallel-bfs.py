import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import random
import sys
import pickle

from sparsenet import longest_shortest_path
from time import time


def parallel_bfs(G: nx.Graph, path: list):
    # starting from the path, we take the neighbhours of all the vertices, moving outward like the waves that form when a stone falls in water.
    # the first wave is the given path; after that, the next wave is the neighbours of the path. Then, take the neighbours of that wave. Repeat until the entire graph is done.
    # this generates the waves so that we don't use up memory storing all the waves
    visited = set(path)
    wave = path
    yield wave, 0
    while len(visited) != G.number_of_nodes():
        start_time = time()
        wave = set(sum((list(G[node].keys()) for node in wave), [])) - visited
        visited |= wave
        yield wave, time() - start_time


if __name__ == '__main__':
    G = nx.Graph()
    G = nx.readwrite.gexf.read_gexf(sys.stdin).to_undirected()

    connected_nodes = sorted(nx.connected_components(G), key=lambda comp: len(comp))[-1]
    component = G.subgraph(connected_nodes)

    path = ['http://www.semanticweb.org/tonka/ontologies/2015/5/tmi-atu-ontology#169B*', 'http://www.semanticweb.org/tonka/ontologies/2015/5/tmi-atu-ontology#169*', 'http://www.semanticweb.org/tonka/ontologies/2015/5/tmi-atu-ontology#Wild_Animals_and_Humans_150–199', 'http://www.semanticweb.org/tonka/ontologies/2015/5/tmi-atu-ontology#157', 'http://www.semanticweb.org/tonka/ontologies/2015/5/tmi-atu-ontology#J17', 'http://www.semanticweb.org/tonka/ontologies/2015/5/tmi-atu-ontology#J', 'http://www.semanticweb.org/tonka/ontologies/2015/5/tmi-atu-ontology#J2073', 'http://www.semanticweb.org/tonka/ontologies/2015/5/tmi-atu-ontology#750A', '2708', 'INF92', '1916', 'http://www.semanticweb.org/tonka/ontologies/2015/5/tmi-atu-ontology#2025', '2898', 'INF72', '2900']
    try:
        with open('pos.pickle', 'rb') as f:
            pos = pickle.load(f)
    except:
        pos = nx.spring_layout(component)
        with open('pos.pickle', 'wb') as f:
            pickle.dump(pos, f)

    os.system("rm -f plots/parallel-bfs-*.png")
    waves = [wave for wave, _ in parallel_bfs(component, path)]
    for i, wave in enumerate(waves):
        nx.draw_networkx_nodes(component, pos, nodelist=wave, node_size=1)
        plt.savefig('plots/parallel-bfs-%s.png'%i)
        plt.clf()
    plt.savefig('plots/parallel-bfs-%s.png'%i)
    for wave in waves:
        r = lambda: random.randint(0,255)
        color = '#%02X%02X%02X' % (r(),r(),r())
        nx.draw_networkx_nodes(component, pos, nodelist=wave, node_size=1, node_color=color)
    plt.savefig('plots/all-waves.png')
    plt.clf()
    os.system("ffmpeg -framerate 1 -i plots/parallel-bfs-%01d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p plots/parallel-bfs.mp4 -y")

    sizes_and_times = np.array([[(len(wave), comp_time) for wave, comp_time in parallel_bfs(component, path)] for _ in range(10)])

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Wave')
    ax1.set_ylabel('Time (seconds)')
    ax1.plot(np.median(sizes_and_times[:, :, 0], axis=0))
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('Size (number of nodes)')  # we already handled the x-label with ax1
    ax2.plot(np.median(sizes_and_times[:, :, 1], axis=0))
    ax2.tick_params(axis='y')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    plt.title('Time to compute each wave and size of each wave (median, n=10)')
    plt.savefig('plots/wave-times.png')
