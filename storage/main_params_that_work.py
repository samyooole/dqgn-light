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
    learning_rate = 1e-4, 
    gamma=0.9, 
    epsilon=0.9, 
    epsilon_min=0.1, 
    epsilon_decay=0.92, 
    training_interval= 30, 
    update_interval=60
)

controller.gymroutine(episodes=6*4)

controller.gymroutine(episodes=6)
controller.workout()