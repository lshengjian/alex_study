import time
from collections import deque
from typing import  Any
import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium.vector.vector_env import  VectorEnv
from gymnasium.vector.vector_env import  VectorWrapper
import numpy as np


class RecordEpisodeStatistics(VectorWrapper):
    def __init__(
        self,
        env: VectorEnv,
        buffer_length: int = 100,
        stats_key: str = "episode",
    ):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            buffer_length: The size of the buffers :attr:`return_queue`, :attr:`length_queue` and :attr:`time_queue`
            stats_key: The info key to save the data
        """
        super().__init__(env)
        self._stats_key = stats_key

        self.n_traj = env.unwrapped.n_traj
        #self.t0 = time.perf_counter()
        self.episode_count = 0

        # self.episode_start_times: np.ndarray = np.zeros(())
        # self.episode_returns: np.ndarray = np.zeros(())
        # self.episode_lengths: np.ndarray = np.zeros((), dtype=int)
        # self.prev_dones: np.ndarray = np.zeros((), dtype=bool)

        self.time_queue = deque(maxlen=buffer_length)
        self.return_queue = deque(maxlen=buffer_length)
        self.length_queue = deque(maxlen=buffer_length)

    def reset(
        self,
        seed: int  = None,
        options: dict  = None,
    ):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = super().reset(seed=seed, options=options)

        self.episode_start_times = np.full((self.num_envs, ), time.perf_counter())
        self.episode_returns = np.zeros((self.num_envs, self.n_traj))
        self.episode_lengths = np.zeros((self.num_envs, ), dtype=int)
        self.prev_dones = np.zeros((self.num_envs,), dtype=bool)
        #self.episode_count = 0
        print('prev_dones',self.prev_dones)


        return obs, info

    def step(
        self, actions: ActType
    ) :
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(actions)

        assert isinstance(
            infos, dict
        ), f"`vector.RecordEpisodeStatistics` requires `info` type to be `dict`, its actual type is {type(infos)}. This may be due to usage of other wrappers in the wrong order."

        
        self.episode_returns[self.prev_dones,:] = 0
        self.episode_lengths[self.prev_dones] = 0
        self.episode_start_times[self.prev_dones] = time.perf_counter()
        # print(self.prev_dones)
        # print(rewards[~self.prev_dones])
        # print(self.episode_returns[~self.prev_dones,:])
        self.episode_returns[~self.prev_dones,:] += rewards[~self.prev_dones]
        self.episode_lengths[~self.prev_dones] += 1

        self.prev_dones = dones = np.logical_or(terminations, truncations)
        
        num_dones = np.sum(dones)

        if num_dones:
            if self._stats_key in infos or f"_{self._stats_key}" in infos:
                raise ValueError(
                    f"Attempted to add episode stats when they already exist, info keys: {list(infos.keys())}"
                )
            else:
                episode_time_length = np.round(
                    time.perf_counter() - self.episode_start_times, 6
                )
                infos[self._stats_key] = {
                    "r": np.where(dones, self.episode_returns[:,-1],0),
                    "l": np.where(dones, self.episode_lengths, 0),
                    "t": np.where(dones, episode_time_length, 0.0),
                }
                infos[f"_{self._stats_key}"] = dones

            self.episode_count += num_dones

            for i in np.where(dones):
                self.time_queue.extend(episode_time_length[i])
                self.return_queue.extend(self.episode_returns[i])
                self.length_queue.extend(self.episode_lengths[i])

        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )


    '''
    def __init__(self, env: gym.Env, deque_size=100):
        super().__init__(env)
        gym.utils.RecordConstructorArgs.__init__(
            self, deque_size=deque_size
        )
        gym.Wrapper.__init__(self, env)
        self.num_envs = getattr(env, "num_envs", 1)#??
        self.n_traj = env.n_traj
        self.t0 = time.perf_counter()
        self.episode_count = 0
        self.episode_returns = None
        self.episode_lengths = None
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
        ) -> tuple[WrapperObsType, dict[str, Any]]:

        self.episode_returns = np.zeros((self.num_envs, self.n_traj), dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.finished = [False] * self.num_envs
        return super().reset(seed=seed, options=options)

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += rewards
        self.episode_lengths += 1
        if not self.is_vector_env:
            infos = [infos]
            dones = [dones]
        else:
            infos = list(infos)  # Convert infos to mutable type
        for i in range(len(dones)):
            if dones[i].all() and not self.finished[i]:
                infos[i] = infos[i].copy()
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {
                    "r": episode_return.copy(),
                    "l": episode_length,
                    "t": round(time.perf_counter() - self.t0, 6),
                }
                infos[i]["episode"] = episode_info
                self.return_queue.append(episode_return)
                self.length_queue.append(episode_length)
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
                self.finished[i] = True

        if self.is_vector_env:
            infos = tuple(infos)
        return (
            observations,
            rewards,
            dones if self.is_vector_env else dones[0],
            infos if self.is_vector_env else infos[0],
        )
    '''
