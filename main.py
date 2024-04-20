from GridEnvironment import GridEnvironment
from controllers import FixedTime, MaxPressure,dqgnLight


genv = GridEnvironment(3, 3, 2, 5, no_cars=200)
controller = dqgnLight(genv.num_nodes_phases, 100, 5)

controller.forward(genv.get_numeric_state(), genv.edge_index)

controller.genv.peek_state()
controller.act()
controller.genv.peek_state()



















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
