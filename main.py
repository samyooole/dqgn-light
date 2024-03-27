import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric

class QValueNet(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim):
        super(QValueNet, self).__init__()
        self.conv = GCNConv(input_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

class TrafficSignalController(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim):
        super(TrafficSignalController, self).__init__()
        self.qvalue_net = QValueNet(num_nodes, input_dim, hidden_dim, output_dim)
        
    def forward(self, x, edge_index):
        q_values = self.qvalue_net(x, edge_index)
        return q_values

# Example usage:
num_nodes = 9  # Assuming 9 intersections in a 3x3 grid
input_dim = 5  # Input feature dimension (you need to define this based on your input)
hidden_dim = 64  # Hidden dimension of the GCN layers
output_dim = 2  # Output dimension for Q-values (2 phases)

# Creating synthetic input features and edge index (you need to define your own)
x = torch.randn(num_nodes, input_dim)  # Example input features
import torch_geometric.utils as utils

# Generate a 3x3 grid graph
edge_index = utils.grid(3,3)

# Creating the traffic signal controller
controller = TrafficSignalController(num_nodes, input_dim, hidden_dim, output_dim)

# Forward pass
q_values = controller(x, edge_index)
print("Q-values for both phases:", q_values)