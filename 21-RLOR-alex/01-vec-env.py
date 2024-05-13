import gymnasium as gym
from pprint import pprint
from gymnasium.wrappers.vector import RecordEpisodeStatistics
if __name__ == "__main__":
    envs = gym.make_vec("CartPole-v1", num_envs=3,vectorization_mode='sync')
    envs=RecordEpisodeStatistics(envs)
    obs,infos=envs.reset()
    print(obs.shape)
    end = False
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
