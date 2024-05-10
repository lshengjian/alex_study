import numpy as np
import torch
import gym
from models.attention_model_wrapper import Agent
from wrappers.syncVectorEnvPomo import SyncVectorEnv
from wrappers.recordWrapper import RecordEpisodeStatistics

device = 'cpu'
ckpt_path = './data/tsp-12000.pt'
agent = Agent(device=device, name='tsp').to(device)
agent.load_state_dict(torch.load(ckpt_path,map_location='cpu'))


env_id = 'tsp-v0'
env_entry_point = 'envs.tsp_vector_env:TSPVectorEnv'


gym.envs.register(
    id=env_id,
    entry_point=env_entry_point,
)

def make_env(env_id, seed, cfg={}):
    def thunk():
        env = gym.make(env_id, **cfg)
        env = RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

env_id='tsp'
seed=123
envs = SyncVectorEnv([make_env(env_id, seed + i, dict(n_traj=2)) for i in range(1)])
print(f'observation_space:')
for k,v in envs.observation_space.items():
    print(k,v)
print(f'action_space:{envs.action_space}')

num_steps = 5
trajectories = []
agent.eval()
obs = envs.reset()
print(f'observation size:')
for k,v in obs.items():
    print(k,v.shape)
for step in range(0,num_steps ):#
    # ALGO LOGIC: action logic
    with torch.no_grad():
        action, logits = agent(obs)
    obs, reward, done, info = envs.step(action.cpu().numpy())
    trajectories.append(action.cpu().numpy())
print(f'trajectories :{np.array(trajectories).shape}')
print(np.array(trajectories))