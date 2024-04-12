import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric
from torch import Tensor
from typing import Optional, Tuple

from torch_geometric.utils import coalesce

class TrafficSignalController(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim):
        super(TrafficSignalController, self).__init__()
        self.conv = GCNConv(input_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index): # hyperparameter train later on, maybe
        x = self.conv(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x
    

class GridEnvironment():
    def __init__(self, grid_height, grid_width):

        self.grid_height = grid_height
        self.grid_width = grid_width
        

        self.spawn_grid()

    def spawn_grid(self):
        
        ## FIRST, spawn basic edge_index and pos which is the convention which pytorch_geometric works in
        # edge_index: each column indicates a pair of vertices that form an edge
        # pos is a 2d position (x,y) indicating where in the grid the node is at
        pos = grid_pos(self.grid_height, self.grid_width)
        
        # Spawn edge index separately because utils.grid spawns triangular mesh grids, not square grids like we want ):
        edge_index = grid_index(self.grid_height, self.grid_width)

        n_nodes = self.grid_height * self.grid_width

        # build adjacency list to avoid redundant queries
        adjlist = self.build_adjacency_list(edge_index, n_nodes)


        ## SECOND, get all possible source-target pairs, then group source-targets that do not interfere with each other and consider them as a singular phase 
        for node in range(n_nodes):
            pass

    def build_adjacency_list(self, edge_index, n_nodes):
        adjacency_list = [[] for _ in range(n_nodes)]
        for src, dst in edge_index.t().tolist():
            if dst not in adjacency_list[src]:
                adjacency_list[src].append(dst)
            if src not in adjacency_list[dst]:  # Add this line if the graph is undirected
                adjacency_list[dst].append(src)
        return adjacency_list

def grid_pos( # this goes with the above class
    height: int,
    width: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:

    dtype = torch.float if dtype is None else dtype
    x = torch.arange(width, dtype=dtype, device=device)
    y = (height - 1) - torch.arange(height, dtype=dtype, device=device)

    x = x.repeat(height)
    y = y.unsqueeze(-1).repeat(1, width).view(-1)

    return torch.stack([x, y], dim=-1)


def grid_index( # this also goes with the above class thenku
    height: int,
    width: int,
    device: Optional[torch.device] = None,
) -> Tensor:

    w = width
    kernel = [-w, -1, 1, w]
    kernel = torch.tensor(kernel, device=device)

    row = torch.arange(height * width, dtype=torch.long, device=device)
    row = row.view(-1, 1).repeat(1, kernel.size(0))
    col = row + kernel.view(1, -1)
    row, col = row.view(height, -1), col.view(height, -1)
    index = torch.arange(1, row.size(1) - 1, dtype=torch.long, device=device)
    row, col = row[:, index].view(-1), col[:, index].view(-1)

    mask = (col >= 0) & (col < height * width)
    row, col = row[mask], col[mask]

    edge_index = torch.stack([row, col], dim=0)
    edge_index = coalesce(edge_index, num_nodes=height * width)
    return edge_index

# Example usage:
num_nodes = 9  # Assuming 9 intersections in a 3x3 grid: later generalize to road design
input_dim = 5  # Input feature dimension: in the case of roads, this will be the number of phases (eg. people from left side trying to turn right)
hidden_dim = 64  # Hidden dimension of the GCN layers
output_dim = 2  # Output dimension for Q-values (2 phases): this has to be the same as the input dimension lol

# Creating synthetic input features and edge index (you need to define your own)
x = torch.randn(num_nodes, input_dim)  # Example input features

import torch_geometric.utils as utils

# Generate a 3x3 grid graph
_, edge_index = utils.grid(3,3)

# Creating the traffic signal controller
controller = TrafficSignalController(num_nodes, input_dim, hidden_dim, output_dim)

# Forward pass
q_values = controller(x, edge_index)
print("Q-values for both phases:", q_values)

