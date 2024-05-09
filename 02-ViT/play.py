import torch

# # 创建一个示例的张量
# tensor = torch.zeros(3, 3,dtype=float)

# # 定义要散射的值
# values = torch.tensor([[1, 2, 3],
#                         [4, 5, 6],
#                         [7, 8, 9]],dtype=float)

# # 定义要散射的索引
# index = torch.tensor([[0, 1, 2],
#                        [2, 0, 1]])

# # 使用scatter_函数将值散射到张量中
# tensor.scatter_(dim=0, index=index, src=values)

# print("散射后的张量：")
# print(tensor)
mask=torch.tensor([1]*3)
chosen_idx=torch.tensor([0,2])
print(mask.scatter_(0, chosen_idx, 0))
mask=torch.tensor([1,1,1]*2)
chosen_idx=torch.tensor([[0,2],[1,2]])#.unsqueeze(1)

print(mask.scatter_(1, chosen_idx, 0))