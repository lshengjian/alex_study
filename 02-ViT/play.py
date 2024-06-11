import torch

# # 创建一个示例的张量
data = torch.zeros(3, 3,dtype=float)

values = torch.tensor([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]],dtype=float)


index = torch.tensor([[0, 1, 2],
                       [2, 0, 1]])



data.scatter_(dim=0, index=index, src=values)
#data[0][0]=value[0][0] data[1][1]=value[1][1]  data[2][2]=value[2][2] 
#data[2][0]=value[2][0] data[0][1]=value[0][1]  data[1][2]=value[1][2]
'''
index[0]是[0, 1, 2]-->values[0]即[1, 2, 3]被复制到data的第一行。
index[1]是[2, 0, 1]-->values[1]即[4, 5, 6]中的元素被复制到data的第二行的第2、0、1位置，即[7, 5, 6]。
'''
print("散射后的张量：")
print(data)
