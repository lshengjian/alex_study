
import matplotlib.pyplot as plt

from rl4co.envs import FJSPEnv

from rl4co.models.zoo.hetgnn.encoder import HetGNNEncoder
from pprint import pprint
from rl4co.utils.decoding import random_policy, rollout

generator_params = {
  "num_jobs": 3,  # the total number of jobs
  "num_machines": 4,  # the total number of machines that can process operations
  "min_ops_per_job": 2,  # minimum number of operatios per job
  "max_ops_per_job": 4,  # maximum number of operations per job
  "min_processing_time": 5,  # the minimum time required for a machine to process an operation
  "max_processing_time": 9,  # the maximum time required for a machine to process an operation
  "min_eligible_ma_per_op": 1,  # the minimum number of machines capable to process an operation
  "max_eligible_ma_per_op": 2,  # the maximum number of machines capable to process an operation
}

env = FJSPEnv(generator_params=generator_params)
td = env.reset(batch_size=[1])
encoder = HetGNNEncoder(embed_dim=32, num_layers=2)
(ma_emb, op_emb), init = encoder(td)
reward, td, actions = rollout(env, td, random_policy)
pprint(actions[0].numpy())
env.render(td, 0)
plt.show()
