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
import threading
import time

from dash.dependencies import Input, Output
from graph.plot_graph import plot_graph
from graph.parallel_bfs import parallel_bfs
from graph.sparsenet import sparsenet

# let's get the sparsenet.
# for simplicity, we'll treat this as a blackbox.
# the only thing we need to know is that we get sparsenet_graph and sparsenet_fig_params at the end
G = nx.read_graphml('../data/combined.graphml')
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

app = dash.Dash(external_stylesheets=['https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css'])
app.layout = html.Div(children=[
    html.Link(href='./index.css', rel='stylesheet'),
    html.Div(className='container', children=[
        html.Div(id='topbar', className='row', children=[
            html.Div(className='three columns', children=''),
            html.Div(className='six columns', children=[
                html.Div(id='graph-description'),
            ]),
            html.Div(className='three columns'),
        ]),
        html.Div(className='row', children=[
            html.Div(className='three columns', children=[
                html.Label('Frame of leaf-originated BFS'),
                daq.NumericInput(id='bfs-frame', value=0, min=0, max=len(bfs_from_leaves_frames)),
                html.Label('First path in SparseNet to include'),
                daq.NumericInput(id='sparsenet-start-path', value=0, min=0, max=10 ** 6),
                html.Label('Last path in SparseNet to include'),
                daq.NumericInput(id='sparsenet-end-path', value=1, min=1, max=10 ** 6),
                dcc.Checklist(id='enable-relayout', options=[{'label': 'Continue to relayout', 'value': 'relayout'}, ], value=[]),
            ]),
            html.Div(className='six columns', children=[
                cyto.Cytoscape(id='cytoscape', elements=[], layout={'name': 'preset'}),
            ]),
            html.Div(className='three columns', children=[
                dcc.Graph(id='path-lengths'),
            ]),
        ]),
        dcc.Interval(id='relayout', interval=1000, n_intervals=0)
    ])
])


def find_sparsenet_in_background():
    for path in sparsenet_generator:
        configuration.append(path)
        
        number_of_edges = sum(len(path) - 1 for path in configuration)
        number_of_nodes = len(set(sum(configuration, [])))
        print(number_of_nodes, number_of_edges, len(configuration))
        if number_of_edges % (2 ** 10) == 0:
            sparsenet_graph = G.subgraph(sum(configuration, []))
            positions[len(configuration)] = nx.spring_layout(sparsenet_graph)


find_sparsenet_in_background_thread = threading.Thread(target=find_sparsenet_in_background)
find_sparsenet_in_background_thread.daemon = True
find_sparsenet_in_background_thread.start()


@app.callback([Output('cytoscape', 'elements'), Output('path-lengths', 'figure'), Output('relayout', 'max_intervals'), Output('graph-description', 'children')], 
              [Input('bfs-frame', 'value'), Input('sparsenet-start-path', 'value'), Input('sparsenet-end-path', 'value'), Input('relayout', 'n_intervals'), Input('enable-relayout', 'value')])
def render_interactive_graph_plot(n, starting_path, ending_path, n_intervals, enable_relayout):
    enable_relayout = 'relayout' in enable_relayout
    ending_path = min(ending_path, 100 * (ending_path // 100 + 1))

    sparsenet_graph = G.subgraph(sum(itertools.islice(configuration, starting_path, ending_path), [])).copy()
    
    if enable_relayout:
        positions[100 * (ending_path // 100 + 1)] = nx.spring_layout(
            G.subgraph(sum(itertools.islice(configuration, 0, 100 * (ending_path // 100 + 1)), [])), 
            pos=positions[100 * (ending_path // 100 + 1)], iterations=10)
        
    pos = {node:position for node, position in positions[100 * (ending_path // 100 + 1)].items() if node in sparsenet_graph.nodes()}

    nodes = [{'data': {'id': node_id}, 'position': { 'x': x * 1000, 'y': y * 1000 }} 
             for node_id, (x, y) in pos.items()]
    edges = [{'data': {'source': src, 'target': dest}} for src, dest in sparsenet_graph.edges()]

    path_lengths = pd.Series(len(path) for path in configuration).value_counts()
    path_lengths = pd.DataFrame({'length': path_lengths.index, 'freq': path_lengths}).sort_index()
    
    graph_description = [
        html.Strong('Edges'),
        html.P(f'|V|: {str(len(nodes))}, |E|: {str(len(edges))}')
    ]

    return nodes + edges, px.line(path_lengths, x="length", y="freq", title='Number of paths of each length'), -1 if enable_relayout else 0, graph_description

app.run_server(debug=True, host='ilab.cs.rutgers.edu', port=4405)