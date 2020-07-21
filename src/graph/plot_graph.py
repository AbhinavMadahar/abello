import math
import networkx as nx
import numpy as np
import plotly.graph_objects as go

def plot_graph(G: nx.Graph):
    pos = nx.spring_layout(G, iterations=200)
    for node in G.nodes:
            G.nodes[node]['pos'] = list(pos[node]) 
    
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
                titleside='right',
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
           margin=dict(b=0,l=0,r=0,t=0),
           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    }
    
    return figure_params