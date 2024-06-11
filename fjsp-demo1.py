import torch
import matplotlib.pyplot as plt

from rl4co.envs import FJSPEnv

from rl4co.models.zoo.hetgnn.encoder import HetGNNEncoder
from rl4co.models.zoo.hetgnn.decoder import HetGNNDecoder

from pprint import pprint

NUM_BATCH=2
NUM_JOBS=3
NUM_MACHINES=4
EMBED_DIM=32
MAX_OPS_PER_JOB=3
generator_params = {
  "num_jobs": NUM_JOBS,  # the total number of jobs
  "num_machines": NUM_MACHINES,  # the total number of machines that can process operations
  "min_ops_per_job": 2,  # minimum number of operatios per job
  "max_ops_per_job": MAX_OPS_PER_JOB,  # maximum number of operations per job
  "min_processing_time": 5,  # the minimum time required for a machine to process an operation
  "max_processing_time": 9,  # the maximum time required for a machine to process an operation
  "min_eligible_ma_per_op": NUM_MACHINES//3+1,  # the minimum number of machines capable to process an operation
  "max_eligible_ma_per_op": NUM_MACHINES//3+2,  # the maximum number of machines capable to process an operation
}

env = FJSPEnv(generator_params=generator_params)
td = env.reset(batch_size=[NUM_BATCH])
pprint(f"作业开始下标:{td['start_op_per_job'][0]}")
pprint(f"作业结束下标:{td['end_op_per_job'][0]}")
assert td['busy_until'].shape==(NUM_BATCH,NUM_MACHINES)
assert td['end_op_per_job'].shape==(NUM_BATCH,NUM_JOBS)
assert td['proc_times'].shape==(NUM_BATCH,NUM_MACHINES,NUM_JOBS*MAX_OPS_PER_JOB)

encoder = HetGNNEncoder(embed_dim=EMBED_DIM, num_layers=2)
(ma_emb, op_emb), init = encoder(td)
assert init is None
assert ma_emb.shape==(NUM_BATCH,NUM_MACHINES,EMBED_DIM) #机器嵌入表示
assert op_emb.shape==(NUM_BATCH,NUM_JOBS*MAX_OPS_PER_JOB,EMBED_DIM) #加工嵌入表示 
assert td["next_op"].shape==(NUM_BATCH,NUM_JOBS)

decoder = HetGNNDecoder(embed_dim=EMBED_DIM)
logits, mask = decoder(td, (ma_emb, op_emb), num_starts=0)
assert logits.shape==(NUM_BATCH,1+NUM_JOBS*NUM_MACHINES) # 1 is NO OP: skip cur time step
assert mask.shape==logits.shape

ptimes=td["proc_times"][0].transpose(0, 1)
pprint(f'处理时间：{ptimes}')
for j_id  in range(0,NUM_JOBS):
  msg=f'J{j_id+1} '
  for i in range(MAX_OPS_PER_JOB):
    k=j_id*MAX_OPS_PER_JOB+i
    ts=ptimes[k]
    idxs=ts>0
    if idxs.sum()<1:
      continue
    for idx,flag in enumerate(idxs):
      if flag:
        msg+=f'O{i+1}|M{idx+1}:{ts[idx]} '
  print(msg)

pprint(f'当前加工：{td["next_op"][0]}')

pprint(f"作业当前加工({NUM_JOBS})行X机器({NUM_MACHINES}列)的关系")
pprint(mask[0][1:].view(NUM_JOBS,NUM_MACHINES)) #作业与机器的可用性

data=[]
def make_step(td):
    logits, mask = decoder(td, (ma_emb, op_emb), num_starts=0)
    ls=logits.masked_fill(~mask, -torch.inf)
    action = ls.argmax(1)
    # pprint(ls[0])
    # pprint(action[0])
    td["action"] = action
    # if action[0] :
    #    print(action[0].item())
    data.append(action[0].item())
    td = env.step(td)["next"]
    return td



while not td["done"].all():
    td = make_step(td)
env.render(td, 0)

for d in data:
    idx=d-1
    if idx>=0:
      print(f"J{idx//4} in M{idx%4}")

plt.show()