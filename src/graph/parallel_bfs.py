import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import random
import sys
import pickle

def parallel_bfs(G: nx.Graph, path: list):
    # starting from the path, we take the neighbhours of all the vertices, moving outward like the waves that form when a stone falls in water.
    # the first wave is the given path; after that, the next wave is the neighbours of the path. Then, take the neighbours of that wave. Repeat until the entire graph is done.
    # this generates the waves so that we don't use up memory storing all the waves
    visited = set(path)
    wave = path
    yield wave
    nodes = set(G.nodes())
    while len(wave):
        wave = set(sum((list(G[node].keys()) for node in wave if node in nodes), [])) - visited
        visited |= wave
        yield wave
