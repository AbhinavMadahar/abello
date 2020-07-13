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

def plot_graph(G: nx.Graph, output_file: str = None, calculate_pos:bool =False):
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

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='<br>Network graph made with Python',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    if output_file:
        fig.write_html(output_file, auto_open=True)
    
    return fig


# First, we need to get the distance matrix.

# In[ ]:


connected_component = G.subgraph(sorted(list(nx.connected_components(G)), key=lambda comp: len(comp))[-1])
distance_matrix = np.load('distance_matrices/component-0.npy')
vertex_name_to_index = { node:i for i, node in enumerate(connected_component.nodes) }


# Ok, now let's get the SparseNet.

# In[ ]:


print('Finding configuration')
configuration = []
for path in sparsenet(connected_component, distance_matrix, vertex_name_to_index):
    print(len(path))
    if len(path) > 6:
        configuration.append(path)
    else:
        break
print('Configuration found')


# Ok, now we can plot the SparseNet.

# In[ ]:

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "wide-form" data frame with no index
# see https://plotly.com/python/wide-form/ for more options
df = pd.DataFrame({"x": [1, 2, 3], "SF": [4, 1, 2], "Montreal": [2, 4, 5]})

fig = px.bar(df, x="x", y=["SF", "Montreal"], barmode="group")

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='sparsenet',
        figure=plot_graph(G.subgraph(sum((path for path in configuration if len(path) >= 6), [])), None, calculate_pos=True)
    )
])

print('Running server...')
app.run_server(debug=True, host='ilab.cs.rutgers.edu', port=4405)