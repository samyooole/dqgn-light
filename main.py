from GridEnvironment import GridEnvironment
from controllers import FixedTime, MaxPressure,dqgnLight

# seal this up!!
genv = GridEnvironment(3, 3, 2, 5, no_cars=500, total_timesteps=200)
# this works kinda nice
controller = dqgnLight(
    gridenvironment = genv,
    num_nodes_phases = genv.num_nodes_phases, 
    hidden_dim= 100, 
    batch_size = 20, 
    learning_rate = 1e-9,  # maybe it's the learning rate after all?
    gamma=0.9, 
    epsilon=0.9, 
    epsilon_min=0.1, 
    epsilon_decay=0.9, 
    training_interval= 30, 
    update_interval=60
)


controller.gymroutine(episodes=10)

mrw = list(controller.erBuffer)[-199:]


genv = GridEnvironment(3, 3, 2, 5, no_cars=500)
controller = FixedTime(genv, genv.num_nodes_phases)
controller.act()
controller.genv.get_numeric_state()
lala+=1


####

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
