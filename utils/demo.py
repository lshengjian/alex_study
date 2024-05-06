import torch
actions=torch.tensor([[0,2,1],[2,1,0]])
d_new=actions.data.new()
data=torch.arange(actions.size(1), out=d_new).view(1, -1).expand_as(actions)
print(data)
print((data==actions.data.sort(1)[0]).all())

#             
#             
#             == actions.data.sort(1)