import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_3d_scalar(data, x_lim, y_lim, z_lim):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    values = data[:, 3]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=values,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title='log(Concentration)')
        )
    )])

    fig.update_layout(
        title='3D Concentration map',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(range=[0, x_lim]),  
            yaxis=dict(range=[0, y_lim]),  
            zaxis=dict(range=[0, z_lim])
        ),
        width=1000,
        height=800
    )
    return fig



def plot_3d_vector(data):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    values = data[:, 3]
    cbarlocs = [.3, .66, .99]
    fig = make_subplots(rows=1, cols=3,
                        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}]])
    fig.add_trace(
        go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=values,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title='value_1', x= cbarlocs[0])
        )
    ), row=1,col=1
    )
    values = data[:, 4]
    fig.add_trace(
        go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=values,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title='value_2', x= cbarlocs[1])
        )
    ), row=1,col=2
    )
    values = data[:, 5]
    fig.add_trace(
        go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=values,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title='value_3', x= cbarlocs[2])
        )
    ), row=1,col=3
    )
    # fig = go.Figure(data=[])

   

    return fig
