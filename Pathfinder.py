import gym
import numpy as np  
from gym.envs.toy_text.frozen_lake import generate_random_map
from gym.envs.registration import register
# Constants
ALPHA = 0.8 
GAMMA = 0.95
epsilon = 1.0                
max_epsilon = 1.0             
min_epsilon = 0.01            
decay_rate = 0.001 
## 

def map_update(custom_map, pc, update_value):
    ls = []
    updated = []
    x,y = pc
    for states in custom_map:
        split = [char for char in states]
        ls.append(split)

    ls = np.array(ls)
    try:
        ls[x][y] = update_value
    except:
        print("Out of bounds error")
        return
    ## Convert array to map format
    
    for item in ls:
        updated.append("".join(item))
    
    return updated  

def generate_map(flag=False, **kwargs):
    
    if flag:
        passmap = map_update(kwargs['passmap'],kwargs['start'],np.random.choice(['H','F'], size=1, p=[0.5,0.5])[0])
        passmap = map_update(passmap,kwargs['goal'], np.random.choice(['H','F'], size=1, p=[0.5,0.5])[0])
        
        return passmap
    
    random_map = generate_random_map(size=kwargs['size'], p=kwargs['prob'])
    
    random_map = map_update(random_map,(0,0),np.random.choice(['H','F'], size=1, p=[0.5,0.5])[0])
    random_map = map_update(random_map,(-1,-1),np.random.choice(['H','F'], size=1, p=[0.5,0.5])[0])
    return random_map

def fix_s_and_d(custom_map, source_cd, destination_cd=(-1,-1)):
        
    new_map = map_update(custom_map,source_cd,'S')
    new_map = map_update(new_map,destination_cd,'G')
    return new_map, source_cd

def prepare_env(new_map, name, episode_count, reward_thresh=0.8196):
    
    try:
        register(
            id=name, 
            entry_point='gym.envs.toy_text:FrozenLakeEnv',
            kwargs={'is_slippery': False},
            max_episode_steps=100,
            reward_threshold=.8196
        )
    except:
           print("Already registered")
            
def epsilon_greedy_action_selection(epsilon, q_table, discrete_state,env):

    random_number = np.random.random()
    if random_number > epsilon: 
        state_row = q_table[discrete_state,:]
        action = np.argmax(state_row) 
   
    else:
        action = env.action_space.sample()
        
    return action

def compute_next_q_value(old_q_value, reward, next_optimal_q_value):
    
    return old_q_value +  ALPHA * (reward + GAMMA * next_optimal_q_value - old_q_value)


def reduce_epsilon(epsilon,epoch):
    
    return min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*epoch)

def rl_agent(custom_map):
    name = "New"+f'-v{np.random.randint(0,1000000,1)[0]}'
    prepare_env(custom_map,name,100)
    env = gym.make(name,desc=custom_map)
    return env

def train(env,epochs):
      
    EPOCHS=epochs 
    action_size = env.action_space.n
    state_size = env.observation_space.n
    q_table = np.zeros([state_size, action_size])
    rewards = []
   
    for episode in range(EPOCHS):
   
        state = env.reset()
        done = False
        total_rewards = 0

        while not done:
            action = epsilon_greedy_action_selection(env, epsilon,q_table, state,env)
            new_state, reward, done, info = env.step(action)
            old_q_value =  q_table[state,action]  
            next_optimal_q_value = np.max(q_table[new_state, :])  
            next_q = compute_next_q_value(old_q_value, reward, next_optimal_q_value)   
            q_table[state,action] = next_q
            total_rewards = total_rewards + reward
            state = new_state
        episode += 1
        epsilon = reduce_epsilon(epsilon,episode) 
        rewards.append(total_rewards)

    return q_table

def return_cd(env, q_table):
    state = env.reset()
    rewards = 0
    u  = [(x,y) for x in range(int((np.sqrt(env.observation_space.n)))) for y in range(int((np.sqrt(env.observation_space.n))))]
    idxtocd = {idx:(x,y) for idx,(x,y) in enumerate(u)}
    cds = []
    
    for _ in range(100):
        cds.append(idxtocd[state])
        action = np.argmax(q_table[state]) 
        state, reward, done, info = env.step(action) 
        if done:
            break

    cds.append(idxtocd[list(idxtocd)[-1]])
    env.close()
    
    return cds

def drone_update(source_cd, drone_cd,obstacle_cd,custom_map):

    new_map = generate_map(True,start = source_cd, goal = (-1,-1), passmap = custom_map)

    # Get the new coordinates from the drone where it detects a potential obstacle  (source_cd) and mark the obstacle in the new map
    new_map = map_update(new_map,obstacle_cd,'H')

    # Fix the new start and the goal positions

    new_map = fix_s_and_d(new_map,drone_cd)

    # Train the agent once again using the same policy as seen before for the same number of EPOCHS
    env= rl_agent(custom_map)
    action_size = env.action_space.n
    state_size = env.observation_space.n
    q_table = np.zeros([state_size, action_size])
    q_table = train(env,200000)
    l = return_cd(env,q_table)
    return l

def main():
    custom_map = generate_map(False,size = 5,prob = 0.5)
    custom_map,source_cd = fix_s_and_d(custom_map,(2,1))
    env= rl_agent(custom_map)
    q_table = train(env,200000)
    
if __name__=='__main__':
    main()