#!/usr/bin/env python
# coding: utf-8

# # SparseNet
# 
# Abhinav Madahar <abhinav.madahar@rutgers.edu> <br />
# James Abello
# 
# ---
# 
# This notebook plots the SparseNet.

# In[ ]:


import math
import networkx as nx
import numpy as np
import pickle
import plotly.graph_objects as go
import random as R

from sparsenet import sparsenet


# Ok, let's load in the fabula graph.

# In[ ]:


G = nx.read_graphml('fabula/combined.graphml')


# To plot the graph, we need to figure out where to put each vertex.
# Instead of recalculating this every time, we just load the saved positions.

# In[ ]:


with open('pos.pickle', 'rb') as pos_file:
    pos = pickle.load(pos_file)
for node in G.nodes:
    G.nodes[node]['pos'] = list(pos[node]) if node in pos else (R.random() * 2 - 1, R.random() * 2 - 1)


# Ok, now let's plot it with Plotly.

# In[ ]:


source_files = ['fabula/atu_only.gexf', 'fabula/mi_only.gexf']
vertex_types = [set(nx.read_gexf(source_file).nodes) for source_file in source_files]

def vertex_type(vertex):
    for i, classification in enumerate(vertex_types):
        if vertex in classification:
            return i+1
    return 0

def plot_graph(G: nx.Graph, output_file: str = None, calculate_pos:bool = False):
    if calculate_pos:
        pos = nx.drawing.layout.spring_layout(G)
        for node in G.nodes:
            G.nodes[node]['pos'] = list(pos[node]) if node in pos else (R.random() * 2 - 1, R.random() * 2 - 1)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        hoverinfo='none',
        mode='lines')

    node_x = [G.nodes[node]['pos'][0] for node in G.nodes()]
    node_y = [G.nodes[node]['pos'][1] for node in G.nodes()]
    labels = [G.nodes[node]['label'] for node in G.nodes()]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=labels,
        marker=dict(
            showscale=True,
            colorscale=[
                [0, 'red'],
                [0.5, 'blue'],
                [1, 'orange']
            ],
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    
    degrees = np.array([math.log(len(adjacencies[1])) for adjacencies in G.adjacency()])

    node_trace.marker.size = degrees
    node_trace.marker.size = (node_trace.marker.size - node_trace.marker.size.min()) / (node_trace.marker.size.max() - node_trace.marker.size.min())
    node_trace.marker.size = node_trace.marker.size * 20 + 5

    colors = []
    for label in node_trace.text:
        if label[:3] == 'ATU':
            color = 1
        elif label[:3] == 'TMI':
            color = 2
        else:
            color = 0
        colors.append(color)
    node_trace.marker.color = colors

    figure_params = {
        'data': [edge_trace, node_trace],
        'layout': go.Layout(
           title='Network graph made with Python',
           titlefont_size=16,
           showlegend=False,
           hovermode='closest',
           margin=dict(b=20,l=5,r=5,t=40),
           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    }
    fig = go.Figure(**figure_params)
    if output_file:
        fig.write_html(output_file, auto_open=True)
    
    return figure_params, fig


# First, we need to get the distance matrix.

# In[ ]:


connected_component = G.subgraph(sorted(list(nx.connected_components(G)), key=lambda comp: len(comp))[-1])
distance_matrix = np.load('distance_matrices/component-0.npy')
vertex_name_to_index = { node:i for i, node in enumerate(connected_component.nodes) }


# Ok, now let's get the SparseNet.

# In[ ]:


configuration = []
for path in sparsenet(connected_component, distance_matrix, vertex_name_to_index):
    print(len(path))
    if len(path) < 6:
        break
    configuration.append(path)


# Ok, now we can plot the SparseNet.

# In[ ]:

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "wide-form" data frame with no index
# see https://plotly.com/python/wide-form/ for more options
df = pd.DataFrame({"x": [1, 2, 3], "SF": [4, 1, 2], "Montreal": [2, 4, 5]})

sparsenet_fig_params, sparsenet_fig = plot_graph(G.subgraph(sum((path for path in configuration if len(path) >= 6), [])))

app.layout = html.Div(children=[
    html.H1(children='SparseNet'),

    html.Div(children='''
        Made by Abhinav Madahar, Fatima Al-Saadeh, Die Hu, and Prof James Abello Monedero.
    '''),

    dcc.Graph(
        id='sparsenet',
        figure=sparsenet_fig
    ),
    
    dcc.Interval(
        id='interval-component',
        interval=1000,
        n_intervals=0
    ),
    
    html.Div(id='live-update-text')
])

import json

@app.callback(Output('sparsenet', 'figure'),
              [Input('sparsenet', 'clickData')])
def update_metrics(clickData):
    try:
        sizes = sparsenet_fig_params['data'][1].marker.size
        point = clickData['points'][0]['pointIndex']
        sizes = sizes + 2 * sizes * np.eye(len(sizes))[point]
        sparsenet_fig_params['data'][1].marker.size = sizes
    except TypeError:
        pass
    return go.Figure(**sparsenet_fig_params)

print('Running server...')
app.run_server(debug=True, host='ilab.cs.rutgers.edu', port=4405)