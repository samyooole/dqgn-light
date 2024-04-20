import torch
from torch import Tensor
from typing import Optional
import networkx as nx
import numpy as np
import queue
from itertools import chain


class GridEnvironment():
    def __init__(self, grid_height, grid_width, timestep_limit, lane_cap, no_cars):

        self.grid_height = grid_height
        self.grid_width = grid_width
        self.timestep_limit = timestep_limit
        self.lane_cap = lane_cap
        self.spawn_grid()
        # Convert edge index tensor to edge list
        edges = self.edge_index.numpy().T.tolist()

        # Create a directed graph from the edge list
        self.G = nx.DiGraph()
        self.G.add_edges_from(edges)
        self.spawn_journies(no_cars) # also gives the initial state


    def car(self, source, sink):
        # A car, when input a source and sink, finds a path composed of flows. We find the edge paths first, then translate that into desired flows
       

        # Find the shortest path using Dijkstra's algorithm
        shortest_path = nx.dijkstra_path(self.G, source=source, target=sink)

        flow_path = []
        for idx, _ in enumerate(shortest_path):
            if idx >= len(shortest_path)-2:
                break # not continue lol
            this_flow = (shortest_path[idx] , shortest_path[idx+2])
            flow_path.append(this_flow)
        
        return flow_path

    def spawn_journies(self, no_cars):
        random_height = np.random.randint(low=0, high=self.grid_height, size=no_cars)
        random_width = np.random.randint(low=0, high=self.grid_width, size=no_cars)

        sources = random_height*self.grid_height + random_width

        random_height = np.random.randint(low=0, high=self.grid_height, size=no_cars)
        random_width = np.random.randint(low=0, high=self.grid_width, size=no_cars)

        sinks = random_height*self.grid_height + random_width

        sourcesinks = [(sources[i], sinks[i]) for i in range(0, no_cars) if sources[i] != sinks[i]]

        flow_paths = [self.car( item[0],item[1]) for item in sourcesinks]

        flow_paths = [flow_path for flow_path in flow_paths if flow_path != []]

        flow_paths = [{ f"car{i}": flow_paths[i] } for i in range(len(flow_paths))] # you can also use the original flowpaths if this is too confusing

        self.flow_paths = {k: v for d in flow_paths for k, v in d.items()}
        self.state = {}


        # initialize state

        # initialize empty queues for every single flow in the phases_list
        all_flows = []
        for sl1 in self.phases_list:
            for sl2 in sl1:
                for flow in sl2:
                    all_flows.append(flow)

        for flow in all_flows:
            self.state[flow] = queue.Queue()



        # LOOK HERE FOR QUEUE LOGIC!
        
        for car, flow_path in self.flow_paths.items():
            self.state[flow_path[0]].put(( car, 0 ) ) # also change 0 here. it's just to keep track of which part of the path the car is in and to tick it up when they get to change phases

    def peek(self, q):
        items = list(q.queue)
        for item in items:
            print(item)

    def peek_state(self):
        for flow, q in self.state.items():
            print(flow)
            print(self.peek(q))
            print('-----------------')


    def dynamics(self, action: list):
        # Receives a list of the chosen phase for each, then evolves

        chosen_phases = [phase[action[idx]] for idx, phase in enumerate(self.phases_list)]

        green_flows = list(chain(*chosen_phases))
        holding_area = []

        # Step 1: Collect all dequeued cars and their intended next flows
        for flow in green_flows:
            if flow not in self.state.keys():
                continue

            num_dequeued = 0
            while (not self.state[flow].empty()) and (num_dequeued < self.timestep_limit):
                car, step = self.state[flow].get()
                holding_area.append((car, step, self.flow_paths[car][step + 1] if step + 1 < len(self.flow_paths[car]) else None))
                num_dequeued += 1

        # Step 2: Enqueue cars based on lane capacity
        for car, step, next_flow in holding_area:
            if next_flow not in self.state:
                if next_flow is not None:
                    self.state[next_flow] = queue.Queue()

            if next_flow and (self.state[next_flow].qsize() < self.lane_cap):
                self.state[next_flow].put((car, step + 1))
            elif next_flow is None:
                # if next_flow is None, we're done and permanently dequeue the car
                continue
            else:
                # Requeue the car back into the current flow
                self.state[flow].put((car, step))
    

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

            self.pos = pos
            self.edge_index = edge_index
            self.phases_list = phases_list
            self.num_nodes_phases = num_nodes_phases
    

    def build_adjacency_list(self, edge_index, n_nodes):
        adjacency_list = [[] for _ in range(n_nodes)]
        for src, dst in edge_index.t().tolist():
            if dst not in adjacency_list[src]:
                adjacency_list[src].append(dst)
            if src not in adjacency_list[dst]:  # Add this line if the graph is undirected
                adjacency_list[dst].append(src)
        return adjacency_list

    def get_numeric_state(self):
        lol = []
        for tlight in self.phases_list:
            lopil = []
            for phase in tlight:
                phase_inline = 0
                for flow in phase:
                    phase_inline += self.state[flow].qsize()

                lopil.append(phase_inline)
            lol.append(lopil)
        x_list = [torch.tensor(phases, dtype=torch.float32) for phases in lol]
        return x_list
    

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



########################################
def dynamics(self, action: list):
        # Receives a list of the chosen phase for each, then evolves

        # sample action DELETE LATER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        action = [phase[0] for phase in self.phases_list]

        green_flows = list(chain(*action))

        for flow in green_flows:

            if flow not in self.state.keys():
                continue

            # Holding area for dequeued cars
            holding_area = queue.Queue()

           # Dequeue all cars from the current flow
            num_dequeued = 0  # Counter variable to keep track of the number of dequeued cars
            while (not self.state[flow].empty()) and (num_dequeued < self.timestep_limit):
                car, step = self.state[flow].get()
                holding_area.put((car, step))
                num_dequeued += 1  # Increment the counter after dequeuing a car

            # Enqueue cars into the next flow
            while not holding_area.empty():
                car, step = holding_area.get()

                if step + 1 < len(self.flow_paths[car]):  # Check if the car has more flows to go through
                    next_flow = self.flow_paths[car][step + 1]  # The subsequent flow it shall enter
                    
                    
                    if next_flow not in self.state:
                        self.state[ next_flow] = queue.Queue()
                    self.state[next_flow].put((car, step + 1))



        # first, there is a limit on how many cars can go during a particular phase timestep - c_bar
        # so, min(car in phase, c_bar) gets pushed into the next edge accordingly. we also need to know which phase they are entering in the next edge, which would require the calculation of paths before hand? 
        # then, check if phase is full <= p_bar. if there is not enough space, keep the overflow of cars in the current phase.
        # 
        # need some queue logic literally in order to return the next state
        pass