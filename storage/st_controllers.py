import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from queue import Queue
import random
from itertools import chain

import torch.optim as optim
import numpy as np

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
        

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_dim, num_nodes_phases):
        super(QNetwork, self).__init__()
        
        max_features = max(num_nodes_phases)
        
        # GCN layers
        self.conv = GCNConv(1, hidden_dim)
        self.mid_conv1 = GCNConv(hidden_dim, hidden_dim)
        self.mid_conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Linear layers for Q-values
        self.q_value_layers = nn.ModuleList([nn.Linear(hidden_dim, n) for n in num_nodes_phases])
        
    def forward(self, x_list, edge_index):
        x = self.conv(x_list, edge_index)
        x=x.relu()
        x=F.dropout(x, p=0.5, training=True)# change this later when doing the normal implementation
        x = self.mid_conv1(x, edge_index)
        x=x.relu()
        x=F.dropout(x, p=0.5, training=True)# change this later when doing the normal implementation
        x = self.mid_conv2(x, edge_index)
        x=x.relu()
        x=F.dropout(x, p=0.5, training=True)# change this later when doing the normal implementation
        
        
        # Compute Q-values
        q_values = [layer(xi) for xi, layer in zip(x, self.q_value_layers)]
        
        return q_values
    

class dqgnLight(nn.Module):
    def __init__(self, gridenvironment, num_nodes_phases, hidden_dim, batch_size, learning_rate, gamma, epsilon, epsilon_min, epsilon_decay, training_interval, update_interval):
        super(dqgnLight, self).__init__()
        
        self.genv = gridenvironment
        self.num_nodes_phases = num_nodes_phases
        self.traffic_lights = [list(range(n)) for n in num_nodes_phases]
        
        # Main Q-network
        self.q_network = QNetwork(max(num_nodes_phases), hidden_dim, num_nodes_phases)
        
        # Target Q-network
        self.target_q_network = QNetwork(max(num_nodes_phases), hidden_dim, num_nodes_phases)
        
        # Copy initial parameters from q_network to target_q_network
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        # Create experience replay buffer: just a list
        self.erBuffer = []

        # Initialize dqn parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.training_interval = training_interval
        self.update_interval = update_interval
        
    def forward(self, x_list, edge_index):
        return self.q_network(x_list, edge_index)
    
    def act(self):

        """
        If step is a training step, call train()
        Else:
            With probability ε, select a random action
            With probability 1-ε, pick the action with the highest expected future reward according to Q(s,a)
        """

        training = (self.epoch % self.training_interval == 0) & (self.epoch !=0) & (len(self.erBuffer) > self.batch_size)

        if training:
            self.train()

        if not training:

            if random.random() < self.epsilon:
                # select a random action
                action = [random.choice(phases) for phases in [range(tlight) for tlight in self.num_nodes_phases]]
            else:
                # get action with the largest value
                with torch.no_grad():
                    this_state = self.current_pressure()
                    q_values = self.forward(this_state, self.genv.edge_index)
                    action = [torch.argmax(q).item() for q in q_values]

            # execute chosen action (set of traffic light green phases)

            # grab state before doing anything
            before_state = self.current_pressure()
            # calculate reward separately as difference? in pressure 
            before_pressure = self.current_pressure()
            self.genv.dynamics(action) # this edits the characteristics of self.genv directly, so we can access whatever we need from here
            after_pressure = self.current_pressure()

            reward = -(np.array(after_pressure) - np.array(before_pressure))
            reward = reward.reshape(reward.shape[0])
            
            #reward = -(np.array(after_pressure)) # just the reward being the pressure period, not the change

            next_state = self.current_pressure()
            terminal = self.timestep == self.genv.total_timesteps - 2 #?

            add_to_buffer = {'state': before_state, 'action': action, 'reward': reward, 'next_state': next_state, 'terminal':terminal}
            # store in replay buffer
            self.erBuffer.append(add_to_buffer)
            self.timestep+=1
        
        
        # Every update_interval, set target network's parameters to Q network's parameters
        time_to_update = (self.epoch % self.update_interval == 0)

        if time_to_update:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        
        self.epoch+=1

    
    def train(self):
        mini_batch = random.sample(self.erBuffer, self.batch_size)

        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        for idx, transition in enumerate(mini_batch):
            state = transition['state']
            next_state = transition['next_state']
            done = transition['terminal']
            reward = transition['reward']
            state_tensor = torch.FloatTensor(state)
            next_state_tensor = torch.FloatTensor(next_state)

            if not done:
                with torch.no_grad():
                    next_q_values = self.target_q_network(next_state_tensor, self.genv.edge_index)
                    max_next_q_values = torch.stack([torch.max(q) for q in next_q_values])
                    Q_target_value = torch.tensor(reward) + self.gamma * max_next_q_values
            else:
                Q_target_value = torch.tensor(reward, dtype=torch.float)

            nowQ = self.q_network(state_tensor, self.genv.edge_index)
            nowQ = torch.stack([torch.max(q) for q in nowQ])
            loss = loss_fn(nowQ, Q_target_value)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Loss: ' + str(loss.item()))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

       

    def workout(self):
        self.genv.spawn_journies(self.genv.no_cars)

        # reset at the start of every workout session
        self.epoch = 0
        self.timestep = 0
        self.erBuffer = [] # must reset experience replay buffer too

        iteration_steps = self.genv.total_timesteps

        while self.timestep < iteration_steps - 1:
            self.act()
    
    def gymroutine(self, episodes):
        for i in range(episodes):
            self.workout()

    def current_pressure(self):
        
        tl_pressure_list=[]
        for tl_id, traffic_light in enumerate(self.traffic_lights):
            tl_pressure = 0
            phases = self.genv.phases_list[tl_id]

            for phase_id, phase in enumerate(phases):
                for flow in phase:
                    #inward pressure
                    inward_pressure = self.genv.state[flow].qsize()
                    tl_pressure+=inward_pressure
                
            # calculate outpressure for each traffic light by itself, otherwise we will double count
            """
            adjacent_tls = self.genv.edge_index[:, self.genv.edge_index[0] == tl_id][1]
            outward_lanes = [self.mine_for_number_in_phases_list(of_intersection=adjacent_tl, from_intersection=tl_id) for adjacent_tl in adjacent_tls]

            unique_outward_lanes = list((set(chain(*outward_lanes))))

            outward_pressure=0

            for lane in unique_outward_lanes:
                outward_pressure += self.genv.state[lane].qsize()

            tl_pressure-=outward_pressure
            """
            tl_pressure_list.append(tl_pressure)
            
        return torch.tensor(tl_pressure_list, dtype=torch.float).unsqueeze(1)

    def mine_for_number_in_phases_list(self, of_intersection, from_intersection):
        this_int = self.genv.phases_list[of_intersection]
        
        filtered_list = [tup for sublist in this_int for tup in sublist if tup[0] == from_intersection]
        return filtered_list






