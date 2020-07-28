import dash
import dash_core_components as dcc
import dash_cytoscape as cyto
import dash_html_components as html
import dash_daq as daq
import itertools
import json
import math
import netgraph
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
sparsenet_generator = sparsenet(connected_component, distance_matrix, vertex_to_index)
configuration = [next(sparsenet_generator)]
sparsenet_graph = G.subgraph(sum(configuration, [])).copy()
sparsenet_fig_params = plot_graph(sparsenet_graph)

leaves = [src for src, destinations in sparsenet_graph.adjacency() if len(destinations) == 1]
bfs_from_leaves_frames = list(parallel_bfs(sparsenet_graph, leaves))

positions = {}

app = dash.Dash()
app.layout = html.Div(children=[
    html.Label('Frame of leaf-originated BFS'),
    daq.NumericInput(id='bfs-frame', value=0, min=0, max=len(bfs_from_leaves_frames)),
    html.Label('First path in SparseNet to include'),
    daq.NumericInput(id='sparsenet-start-path', value=0, min=0, max=10 ** 6),
    html.Label('Last path in SparseNet to include'),
    daq.NumericInput(id='sparsenet-end-path', value=1, min=1, max=10 ** 6),
    dcc.Checklist(id='enable-relayout', options=[{'label': 'Continue to relayout', 'value': 'relayout'}, ], value=['relayout']),
    html.Div(id='graph-description'),
    cyto.Cytoscape(id='cytoscape', elements=[], layout={'name': 'preset'}),
    dcc.Graph(id='path-lengths'),
    dcc.Interval(id='relayout', interval=1000, n_intervals=0)
])


@app.callback([Output('cytoscape', 'elements'), Output('path-lengths', 'figure'), Output('relayout', 'max_intervals')], 
              [Input('bfs-frame', 'value'), Input('sparsenet-start-path', 'value'), Input('sparsenet-end-path', 'value'), Input('relayout', 'n_intervals'), Input('enable-relayout', 'value')])
def render_interactive_graph_plot(n, starting_path, ending_path, n_intervals, enable_relayout):
    enable_relayout = 'relayout' in enable_relayout
    global configuration
    if ending_path > len(configuration):
        try:
            configuration += itertools.islice(sparsenet_generator, ending_path - len(configuration))
        except StopIteration:
            pass
    sparsenet_graph = G.subgraph(sum(itertools.islice(configuration, starting_path, ending_path), [])).copy()
    if (starting_path, ending_path) not in positions:
        positions[(starting_path, ending_path)] = nx.spring_layout(sparsenet_graph, iterations=10)
    elif enable_relayout:
        positions[(starting_path, ending_path)] = nx.spring_layout(sparsenet_graph, pos=positions[(starting_path, ending_path)], iterations=10)
    pos = positions[(starting_path, ending_path)]
    
    nodes = [{'data': {'id': node_id}, 'position': { 'x': x * 1000, 'y': y * 1000 }} 
             for node_id, (x, y) in pos.items()]
    edges = [{'data': {'source': src, 'target': dest}} for src, dest in sparsenet_graph.edges()]
    
    path_lengths = pd.Series(len(path) for path in configuration).value_counts()
    path_lengths = pd.DataFrame({'length': path_lengths.index, 'freq': path_lengths}).sort_index()
    
    print(-1 if enable_relayout else 0)
    return nodes + edges, px.line(path_lengths, x="length", y="freq", title='Number of paths of each length'), -1 if enable_relayout else 0


@app.callback(Output('graph-description', 'children'), 
              [Input('cytoscape', 'elements')])
def update_number_of_vertices(elements):
    number_of_nodes = len([element for element in elements if 'source' not in element['data']])
    number_of_edges = len([element for element in elements if 'source' in element['data']])
    return html.P(str(number_of_nodes) + ' vertices and ' + str(number_of_edges) + ' edges.')


app.run_server(debug=True, host='ilab.cs.rutgers.edu', port=4405)