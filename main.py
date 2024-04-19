import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric
from torch import Tensor
from typing import Optional, Tuple

from torch_geometric.utils import coalesce


class TrafficSignalController(nn.Module):
    def __init__(self, num_nodes_phases, hidden_dim):
        super(TrafficSignalController, self).__init__()
        self.conv = GCNConv(1, hidden_dim)  # Input dimension is 1 for each node
        
        # Dynamic linear layers for each node
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, n) for n in num_nodes_phases])  # num_nodes_phases is a list containing the number of phases for each node
    
    def forward(self, x_list, edge_index): 
        x = torch.cat(x_list, dim=0).unsqueeze(1)  # Concatenate and add a channel dimension
        x = self.conv(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        outputs = []
        start_idx = 0
        for layer in self.layers:
            end_idx = start_idx + len(x_list[start_idx])
            outputs.append(layer(x[start_idx:end_idx]).squeeze(1))
            start_idx = end_idx
        
        return outputs
    

class GridEnvironment():
    def __init__(self, grid_height, grid_width):

        self.grid_height = grid_height
        self.grid_width = grid_width
        self.spawn_grid()

    def spawn_grid(self):
        
        pos = grid_pos(self.grid_height, self.grid_width)
        edge_index = grid_index(self.grid_height, self.grid_width)

        n_nodes = self.grid_height * self.grid_width
        adjlist = self.build_adjacency_list(edge_index, n_nodes)

        phases_list = []
        num_nodes_phases = []
        for node in range(n_nodes):
            this_adj = adjlist[node]
            phases = []
            
            # i apologize to the gods of coding
            if len(this_adj) == 2:
                phases.append([(this_adj[0], this_adj[1]), (this_adj[1], this_adj[0])])
            elif (len(this_adj) == 3) & (0 <= node <= self.grid_width-1): # if it's a T intersection in the first row
                phases.append([(node-1, node+1), (node+1, node-1)])
                phases.append([(node-1, node+self.grid_width), (node+self.grid_width, node-1)])
                phases.append([(node+self.grid_width, node+1), (node+1, node+self.grid_width)])
            elif (len(this_adj) == 3) & (node % self.grid_width == 0): # if it's a T intersection in the first column
                phases.append([(node-self.grid_width, node+self.grid_width), (node+self.grid_width, node-self.grid_width)]) 
                phases.append([(node-self.grid_width, node+1), (node+1, node-self.grid_width)])
                phases.append([(node+self.grid_width, node+1), (node+1, node+self.grid_width)])
            elif (len(this_adj) == 3) & (n_nodes-self.grid_width <= node <= n_nodes-1): # if it's a T intersection in the last row
                phases.append([(node-1, node+1), (node+1, node-1)]) 
                phases.append([(node-1, node-self.grid_width), (node-self.grid_width, node-1)])
                phases.append([(node-self.grid_width, node+1), (node+1, node-self.grid_width)])
            elif (len(this_adj) == 3) & (node % self.grid_width == self.grid_width-1): # if it's a T intersection in the last column
                phases.append([(node-self.grid_width, node+self.grid_width), (node+self.grid_width, node-self.grid_width)]) 
                phases.append([(node-1, node-self.grid_width), (node-self.grid_width, node-1)])
                phases.append([(node+self.grid_width, node-1), (node-1, node+self.grid_width)])
            elif len(this_adj) == 4:
                phases.append([(node-1, node+1), (node+1, node-1)])
                phases.append([(node-self.grid_width, node+self.grid_width), (node+self.grid_width, node-self.grid_width)])
                phases.append([(node-1, node+self.grid_width), (node+1, node-self.grid_width)])
                phases.append([(node+1, node+self.grid_width), (node-1, node-self.grid_width)])
                phases.append([(node+self.grid_width, node+1), (node-self.grid_width, node-1)])
                phases.append([(node+self.grid_width, node-1), (node-self.grid_width, node+1)])
            
            phases_list.append(phases)
            num_nodes_phases.append(len(phases))

        return pos, edge_index, phases_list, num_nodes_phases
    

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

def grid_index(height, width):
    edge_indices = [] # creds chatgpt
    
    # Horizontal edges
    for i in range(height):
        for j in range(width - 1):
            idx1 = i * width + j
            idx2 = i * width + j + 1
            edge_indices.append((idx1, idx2))
            edge_indices.append((idx2, idx1))  # Reverse direction
    
    # Vertical edges
    for i in range(height - 1):
        for j in range(width):
            idx1 = i * width + j
            idx2 = (i + 1) * width + j
            edge_indices.append((idx1, idx2))
            edge_indices.append((idx2, idx1))  # Reverse direction
    edge_indices_tensor = torch.tensor(edge_indices, dtype=torch.long).T
    return edge_indices_tensor

# Example usage
grid_env = GridEnvironment(grid_height=3, grid_width=3)
pos, edge_index, phases_list, num_nodes_phases = grid_env.spawn_grid()

hidden_dim = 128

# Generate random input tensors for each node
x_list = [torch.rand(n, 1) for n in num_nodes_phases]

model = TrafficSignalController(num_nodes_phases, hidden_dim)
outputs = model(x_list, edge_index)

# Example usage:
num_nodes = 9  # Assuming 9 intersections in a 3x3 grid: later generalize to road design
input_dim = 5  # Input feature dimension: in the case of roads, this will be the number of phases (eg. people from left side trying to turn right)
hidden_dim = 64  # Hidden dimension of the GCN layers
output_dim = 2  # Output dimension for Q-values (2 phases): this has to be the same as the input dimension lol

# Creating synthetic input features and edge index (you need to define your own)
x = torch.randn(num_nodes, input_dim)  # Example input features


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
