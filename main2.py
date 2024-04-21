from GridEnvironment import GridEnvironment
from controllers import FixedTime, MaxPressure,dqgnLight

# seal this up!!
genv = GridEnvironment(3, 3, 2, 5, no_cars=500, total_timesteps=200)
# this works kinda nice
controller = dqgnLight(
    gridenvironment = genv,
    num_nodes_phases = genv.num_nodes_phases, 
    hidden_dim= 150, 
    batch_size = 200, 
    learning_rate = 1e-6, 
    gamma=0.9, 
    epsilon=0.9, 
    epsilon_min=0.1, 
    epsilon_decay=1, 
    training_interval= 300, 
    update_interval=300
)


controller.gymroutine(episodes=100) # let it see the world first, i guess

controller.epsilon_decay=0.92
controller.training_interval=50
controller.update_interval=100
controller.batch_size=200



controller.gymroutine(episodes=1)

mostrecent_workout = list(controller.erBuffer)[-199:]



zeroness = []
for i in tqdm(range(50)):
    controller.workout()
    
    mostrecent_workout = list(controller.erBuffer)[-199:]
    states = [experience['state'] for experience in mostrecent_workout]

    for idx, state in enumerate(states):
        if torch.all(state==0):
            break
    print(idx)
    zeroness.append(idx)
