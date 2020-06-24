import networkx as nx
import sys
from collections import namedtuple


if __name__ == '__main__':
    G = nx.readwrite.gexf.read_gexf(sys.argv[1])
    if type(G) == nx.classes.digraph.DiGraph:
        G = nx.MultiGraph(G)
    for filename in sys.argv[2:]:
        newgraph = nx.readwrite.gexf.read_gexf(filename)
        if type(newgraph) == nx.classes.digraph.DiGraph:
            newgraph = nx.MultiGraph(newgraph)
        G = nx.compose(G, newgraph)
    for line in nx.readwrite.edgelist.generate_edgelist(G):
        print(line)
