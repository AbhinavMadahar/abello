import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import json
import math
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
import random as R

from dash.dependencies import Input, Output
from graph.plot_graph import plot_graph
from graph.parallel_bfs import parallel_bfs
from graph.sparsenet import sparsenet

# let's get the sparsenet.
# for simplicity, we'll treat this as a blackbox.
# the only thing we need to know is that we get sparsenet_graph and sparsenet_fig_params at the end
G = nx.read_graphml('../data/combined.graphml')
with open('cache/pos.pickle', 'rb') as pos_file:
    pos = pickle.load(pos_file)
for node in G.nodes:
    G.nodes[node]['pos'] = list(pos[node]) if node in pos else (R.random() * 2 - 1, R.random() * 2 - 1)
source_files = ['../data/atu_only.gexf', '../data/mi_only.gexf']
vertex_types = [set(nx.read_gexf(source_file).nodes) for source_file in source_files]
connected_component = G.subgraph(sorted(list(nx.connected_components(G)), key=lambda comp: len(comp))[-1])
distance_matrix = np.load('../distance_matrices/component-0.npy')
vertex_to_index = { node:i for i, node in enumerate(connected_component.nodes) }
index_to_vertex = { v:i for i, v in vertex_to_index.items() }
configuration = []
for path in sparsenet(connected_component, distance_matrix, vertex_to_index):
    if len(path) < 6:
        break
    configuration.append(path)
sparsenet_graph = G.subgraph(sum(configuration, [])).copy()
sparsenet_fig_params = plot_graph(sparsenet_graph)

leaves = [src for src, destinations in sparsenet_graph.adjacency() if len(destinations) == 1]
bfs_from_leaves_frames = list(parallel_bfs(sparsenet_graph, leaves))

app = dash.Dash()
app.layout = html.Div(children=[
    dcc.Graph(id='sparsenet', figure=go.Figure(**sparsenet_fig_params), style={'height': '90vh'}),
    daq.NumericInput(id='frame', value=0),
    html.Div(id='live-update-text')
])

original_sparsenet_colors = sparsenet_fig_params['data'][1].marker.color
@app.callback(Output('sparsenet', 'figure'), [Input('frame', 'value')])
def start_bfs_from_node(n):
    frame = bfs_from_leaves_frames[n % len(bfs_from_leaves_frames)]
    highlighted = [i for i, vertex in enumerate(sparsenet_graph.nodes()) if vertex in frame]
    colors = ['white' if i in highlighted else color for i, color in enumerate(original_sparsenet_colors)]
    sparsenet_fig_params['data'][1].marker.color = colors
    return go.Figure(**sparsenet_fig_params)

app.run_server(debug=True, host='ilab.cs.rutgers.edu', port=4405)