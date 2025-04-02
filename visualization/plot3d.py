import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.animation import FuncAnimation
sns.set_theme(context='talk', font = 'serif')

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

def plot_3d_surfaces(data, source_z, save_path):

    sns.set_theme(context='talk', font = 'serif', font_scale=1.9)
    sns.set_style("white", {"grid.color": ".6", "grid.linestyle": ":"})

    data_3d = data[data[:, 3] > 5]
    data_3d[:, 3] = np.log(data_3d[:, 3])
    x = data_3d[:, 0]
    y = data_3d[:, 1]
    z = data_3d[:, 2]
    x_size = len(np.unique(x))
    y_size = len(np.unique(y))

    values = data_3d[:, 3]
    df = pd.DataFrame({'x':x, 'y':y, 'z':z, 'c': values})
    # print(np.min(df["c"]), np.max(df["c"]))
    # df.loc[df["c"] <=0, "c"] = 0
    # df["c"] = np.log1p(df["c"])
    # df.loc[df["c"] <0, "c"]= 0
    # print(np.min(df["c"]), np.max(df["c"]))
    # print(x[np.argmax(df["c"]), y[np.argmax(df["c"])]])
    z_unique = np.unique(df.loc[:, "z"])
    idx = (np.abs(z_unique - source_z)).argmin()
    z_near =  z_unique[idx-15: idx+15][::3]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # ax.scatter(x, y, z, c=values, cmap='viridis', marker='o', alpha=0.8)

    
    for z in z_near:
        mask = (df["z"] == z) & (df["c"] != 0)
        X = df.loc[mask, 'x']
        Y = df.loc[mask, 'y']
        Z = df.loc[mask, 'z']
        C = df.loc[mask, 'c']
        
        ax.scatter(X[::2], Y[::2], Z[::2], c=C[::2], cmap='viridis', marker='o', alpha=1)
        # surf = ax.plot_trisurf(df['x'][df["z"] == z], df['y'][df["z"] == z], df['z'][df["z"] == z], 
        # color= df["c"][df["z"] == z],cmap=plt.cm.viridis, linewidth=0.2)
    # # fig.colorbar( surf, shrink=0.5, aspect=5)
    ax.view_init(30, -60)  

    ax.set_xticks(np.linspace(0, 1500, 3))
    ax.set_yticks(np.linspace(0, 1000, 3))
    ax.set_zticks(np.linspace(0, 500, 2))
    
    ax.set_xlabel('x, м', labelpad=25)
    ax.set_ylabel('y, м', labelpad=50)
    ax.set_zlabel('z, м', labelpad=1)

    ax.tick_params(axis='y', pad=10)
    ax.tick_params(axis='z', pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)

    def update(frame):
        ax.clear()
        for z in z_near:
            mask = (df["z"] == z) & (df["c"] != 0)
            X = df.loc[mask, 'x']
            Y = df.loc[mask, 'y']
            Z = df.loc[mask, 'z']
            C = df.loc[mask, 'c']
            ax.set_xticks(np.linspace(0, 1500, 3))
            ax.set_yticks(np.linspace(0, 1000, 3))
            ax.set_zticks(np.linspace(0, 500, 2))
            ax.set_xlabel('x, м', labelpad=25)
            ax.set_ylabel('y, м', labelpad=50)
            ax.set_zlabel('z, м', labelpad=1)
            ax.scatter(X, Y, Z, c=C, cmap='viridis', marker='o', alpha=0.8)
        ax.view_init(30, -60+frame)
        return fig,

    ani = FuncAnimation(fig, update, frames=range(0, 360, 2))
    ani.save('3d_surfaces' + '.gif', fps=30)
    # fig.colorbar(surf)
    
    
