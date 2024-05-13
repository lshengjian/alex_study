import gymnasium as gym
from pprint import pprint
from envs.my_vector_env import SyncVectorEnv
from wrappers.recordWrapper import RecordEpisodeStatistics
#from gymnasium.wrappers.vector import RecordEpisodeStatistics
def make_env(env_id, cfg={}):
    def thunk():
        env = gym.make(env_id, **cfg)
        return env

    return thunk
if __name__ == "__main__":
    env_id='tsp_v1'
    num_envs=3
    gym.envs.register(
        id=env_id,
        entry_point='envs.tsp_vector_env:TSPVectorEnv',
    )
    envs = SyncVectorEnv(
        [
            make_env(
                env_id,
                cfg={"eval_data": False, "max_nodes": 5, "n_traj": 5},
            )
            for _ in range(num_envs)
        ]
    )
    
    obs,infos=envs.reset(seed=1234)

    assert obs['observations'].shape==(3,5,2) #(5个节点，2个坐标)
    assert obs['action_mask'].shape==(3,5,5) #(5条轨迹,5个节点)

    end = False
    envs=RecordEpisodeStatistics(envs)
    obs,infos=envs.reset(seed=1234)
    pprint(envs.episode_count)
    pprint(envs.episode_start_times)
    while not end:
        actions = envs.action_space.sample()  # agent policy that uses the observation and info
        observations, rewards, terminateds, truncateds, infos = envs.step(actions)
        end = terminateds.any() or truncateds.any()
        pprint(envs.episode_count)

    print('-'*10)
    pprint(envs.episode_start_times)
    pprint(envs.episode_returns)
    pprint(infos)
    
    envs.close()
