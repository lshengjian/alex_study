import torch 

data=torch.ones(2,3)

assert data.shape==(2,3)
data1=data[None, None]
assert data1.shape==(1,1,2,3)
data2=data[None, None,...]
assert data2.shape==(1,1,2,3)
data3=data[...,None]
assert data3.shape==(2,3,1)


cities=torch.rand((5,2))*10
cs1=cities.unsqueeze(1) #cities[:,None,:]
print(cs1.shape)
cs2=cs1.permute(1,0,2)
print(cs2.shape)
dis2=(cs1-cs2)**2
dis2=dis2.sum(-1)
dis=torch.sqrt(dis2)
print(dis)

dis=torch.norm(cs1-cs2, dim=-1)
print(dis)
print(cs1.expand(5,5,2))
print(cs2.expand(5,5,2))
