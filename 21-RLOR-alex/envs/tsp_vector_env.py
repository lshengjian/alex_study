import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType,ActType,SupportsFloat

from .tsp_data import TSPDataset
from typing  import Union
from .comm import *


class TSPVectorEnv(gym.Env):
    def __init__(self, **kwargs):
        self.max_nodes = 50
        self.n_traj = 50
        # if eval_data==True, load from 'test' set, the '0'th data
        self.eval_data = False
        self.eval_partition = "test"
        self.eval_data_idx = 0
        assign_env_config(self, kwargs)
        if self.n_traj>self.max_nodes:
            self.n_traj=self.max_nodes

        obs_dict = {"observations": spaces.Box(low=0, high=1, shape=(self.max_nodes, 2),dtype=np.float32)}
        # obs_dict["action_mask"] = spaces.MultiBinary(
        #     [self.n_traj, self.max_nodes]
        # )  # 1: OK, 0: cannot go
        obs_dict["first_node_idx"] = spaces.MultiDiscrete([self.max_nodes] * self.n_traj)
        obs_dict["last_node_idx"] = spaces.MultiDiscrete([self.max_nodes] * self.n_traj)
        obs_dict["is_initial_action"] = spaces.Discrete(2)

        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = spaces.MultiDiscrete([self.max_nodes] * self.n_traj)
        self.reward_space = None


    def reset(
        self,
        *,
        seed: Union[int,None] = None,
        options:Union[Dict[str, Any], None] = None,
    ) -> tuple[ObsType, Dict[str, Any]]:  # type: ignore
        super().reset(seed=seed,options=options)
        self.info = {}
        self.visited = np.zeros((self.n_traj, self.max_nodes), dtype=bool)
        self.num_steps = 0
        self.last = np.zeros(self.n_traj, dtype=int)  # idx of the first elem
        self.first = np.zeros(self.n_traj, dtype=int)  # idx of the first elem
        self.done = np.zeros((self.n_traj,), dtype=bool)
        self.timeout=np.zeros((self.n_traj,), dtype=bool)
        if self.eval_data:
            self._load_orders()
        else:
            self._generate_orders()
        self.state = self._update_state()
        

        return self.state,self.info

    def _load_orders(self):
        self.nodes = np.array(TSPDataset[self.eval_partition, self.max_nodes, self.eval_data_idx])

    def _generate_orders(self):
        self.nodes = np.random.rand(self.max_nodes, 2)

    #def step(self, action):

    def step(
        self, actions: ActType #(n_traj)
    ) -> tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        self._go_to(actions)  # Go to node 'action', modify the reward
        self.num_steps += 1
        self.state :dict= self._update_state()

        # need to revisit the first node after visited all other nodes
        self.done = (actions == self.first) & self.is_all_visited()

        return self.state, self.reward, self.done,self.timeout, self.info

    # Euclidean cost function
    def cost(self, loc1, loc2):
        return dist(loc1, loc2)

    def is_all_visited(self):
        # assumes no repetition in the first `max_nodes` steps
        return self.visited[:, :].all(axis=1)

    def _go_to(self, destination):
        dest_node = self.nodes[destination]
        if self.num_steps != 0:
            dist = self.cost(dest_node, self.nodes[self.last])
        else:
            dist = np.zeros(self.n_traj)
            self.first = destination

        self.last = destination

        self.visited[np.arange(self.n_traj), destination] = True
        self.reward = -dist

    def _update_state(self):
        obs = {"observations": self.nodes}  # n x 2 array
        #obs["action_mask"] = 
        obs["first_node_idx"] = self.first
        obs["last_node_idx"] = self.last
        obs["is_initial_action"] = self.num_steps == 0
        self._update_mask()
        return obs

    def _update_mask(self):
        # Only allow to visit unvisited nodes
        action_mask = ~self.visited
        # can only visit first node when all nodes have been visited
        flag=self.is_all_visited()
        action_mask[:, self.first] |= flag
        #action_mask[np.arange(self.n_traj), self.first] |= flag
        self.info["action_mask"]=action_mask


