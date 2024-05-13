import gymnasium as gym
from pprint import pprint
import numpy as np
import random
if __name__ == "__main__":
    env_id='tsp_v1'
    gym.envs.register(
        id=env_id,
        entry_point='envs.tsp_vector_env:TSPVectorEnv',
    )
    env = gym.make(env_id,max_nodes=5)
    print(env)
    obs,info=env.reset()
    
    assert obs['observations'].shape==(5,2)

    assert info['action_mask'].shape==(5,5)
    end = False
    while not end:
        masks=info['action_mask']
        
        acts=[]
        for i,mask in enumerate(masks):
            # if i==0:
            #     print(list(mask))
            idxs=np.argwhere(mask).ravel()
            acts.append(random.choice(idxs))
        print(acts)
        #actions = envs.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(acts)
        #pprint(terminated)
        end = terminated.any() #or truncated.any()
   
    pprint(reward)
    pprint(info['action_mask'])
    env.close()
