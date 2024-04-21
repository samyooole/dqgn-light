from GridEnvironment import GridEnvironment
from controllers import FixedTime, MaxPressure,dqgnLight
import torch
from tqdm import tqdm

# seal this up!!
genv = GridEnvironment(3, 3, 2, 5, no_cars=500, total_timesteps=200)
# this works kinda nice
controller = dqgnLight(
    gridenvironment = genv,
    num_nodes_phases = genv.num_nodes_phases, 
    hidden_dim= 100, 
    batch_size = 20, 
    learning_rate = 1e-5, 
    gamma=0.6, 
    epsilon=0.9, 
    epsilon_min=0.1, 
    epsilon_decay=0.92, 
    training_interval= 30, 
    update_interval=60
)

controller.gymroutine(episodes=6)

zeroness = []
for i in tqdm(range(50)):
    controller.workout()
    states = [experience['state'] for experience in controller.erBuffer]

    for idx, state in enumerate(states):
        if torch.all(state==0):
            break
    print(idx)
    zeroness.append(idx)
    





zeroness_ft = []
for i in tqdm(range(50)):
    genv = GridEnvironment(3, 3, 2, 5, no_cars=100)
    controller = FixedTime(genv, genv.num_nodes_phases)
    zeropoint = 0
    while ~torch.all(controller.genv.get_numeric_state()==0):
        controller.act()
        zeropoint+=1
    print(zeropoint)
    zeroness_ft.append(zeropoint)