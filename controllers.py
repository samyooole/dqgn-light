import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from queue import Queue

class FixedTime():
    def __init__(self, gridenvironment, num_nodes_phases):
        self.num_nodes_phases = num_nodes_phases
        self.genv = gridenvironment
        # Create a list of queues
        rnnp = [Queue() for _ in range(len(num_nodes_phases))]

        # Populate each queue with integer values from 0 to num_nodes_phases[idx] - 1
        for idx, num_nodes in enumerate(num_nodes_phases):
            for item in range(num_nodes):
                rnnp[idx].put(item)
        self.rnnp=rnnp
        self.global_counter = 0

    def act(self):
        action = []
        for traffic_light in self.rnnp:
            green_phase = traffic_light.get()
            action.append(green_phase)
            traffic_light.put(green_phase) # put it back into the rotation

        self.genv.dynamics(action)

class MaxPressure():
    def __init__(self, gridenvironment, num_nodes_phases):
        self.num_nodes_phases = num_nodes_phases
        self.genv = gridenvironment
        self.traffic_lights = [list(range(n)) for n in num_nodes_phases]

    def act(self):

        # find the max pressure phase

        action = []
        for tl_id, traffic_light in enumerate(self.traffic_lights):
            phases = self.genv.phases_list[tl_id]

            pressure_dict = {}
            for phase_id, phase in enumerate(phases):
                phase_pressure=0
                for flow in phase:
                    #inward pressure
                    inward_pressure = self.genv.state[flow].qsize()
                    
                    outward_lanes = self.mine_for_number_in_phases_list(of_intersection=flow[1], from_intersection=tl_id)

                    outward_pressure=0

                    for lane in outward_lanes:
                        outward_pressure += self.genv.state[lane].qsize()
                    
                    pressure = inward_pressure-outward_pressure
                    phase_pressure+=pressure
                
                pressure_dict.update({phase_id: phase_pressure})

            MP_phase_id = max(pressure_dict, key=pressure_dict.get)
            action.append(MP_phase_id)

        # act!
        self.genv.dynamics(action)

    def mine_for_number_in_phases_list(self, of_intersection, from_intersection):
        this_int = self.genv.phases_list[of_intersection]
        
        filtered_list = [tup for sublist in this_int for tup in sublist if tup[0] == from_intersection]
        return filtered_list
        
        
class dqgnLight(nn.Module):
    def __init__(self, num_nodes_phases, hidden_dim, m_hops):
        super(dqgnLight, self).__init__()
        
        self.num_nodes_phases = num_nodes_phases
        self.m_hops = m_hops
        max_features = max(num_nodes_phases)
        
        # GCN layers
        self.conv = GCNConv(max_features, hidden_dim)

        # Linear layers for Q-values
        self.q_value_layers = nn.ModuleList([nn.Linear(hidden_dim, n) for n in num_nodes_phases])
        
    def forward(self, x_list, edge_index):
        x = self.conv(x_list, edge_index)
        
        # Compute Q-values
        q_values = [layer(xi) for xi, layer in zip(x, self.q_value_layers)]
        
        return q_values


# Pad tensors to the left
padded_tensors =[F.pad(t, (0, max_features - t.size(0)), 'constant', 0) for t in x_list]


# Create a full tensor
full_tensor = torch.stack(padded_tensors, dim=0)
x_list=full_tensor