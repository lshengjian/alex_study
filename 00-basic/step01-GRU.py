import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 随机生成一些天气数据，例如温度
np.random.seed(0)
batch_size = 64
temperature_data = np.sin(np.arange(0, 100) * 0.1) + \
    np.random.normal(scale=0.01, size=(batch_size,100))

# 转换为PyTorch张量
temperature_tensor = torch.tensor(temperature_data, dtype=torch.float32)

# 标准化数据
mean = temperature_tensor.mean()
std = temperature_tensor.std()
temperature_tensor = (temperature_tensor - mean) / std

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x,h=None):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)#.to(x.device)
        h=h0 if h is None else h
        out, h = self.gru(x, h)
        out = self.fc(out)  
        return out,h

input_size = 1  # 假设输入数据的特征维度为1
hidden_size = 128  # 隐藏层的大小
num_layers = 2  # GRU的层数
output_size = 1  # 输出数据的特征维度

model = GRUModel(input_size, hidden_size, num_layers, output_size)

criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

epochs = 100  # 训练的轮数
seq_length = 10  # 每个序列的长度
h=None
for epoch in range(epochs):
    model.train()
    for i in range(0, len(temperature_tensor[0]) - seq_length, 1):
        inputs = temperature_tensor[:,i:i + seq_length].unsqueeze(-1)#.reshape(batch_size, seq_length, -1)
        targets = temperature_tensor[:,i + 1:i + seq_length + 1].unsqueeze(-1)#.reshape(batch_size, seq_length, -1)
        # if inputs.shape!=targets.shape:
        #     continue
        optimizer.zero_grad()
        outputs,h = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    # 假设我们使用最后一个序列来预测下一个时间步的温度
    last_sequence = temperature_tensor[-5:,:].unsqueeze(-1)
    prediction ,_= model(last_sequence)
    prediction=prediction[0] * std + mean


# # 绘制原始数据和预测结果
plt.figure(figsize=(10, 6))
plt.plot(temperature_data[-5], label='Original Data')
plt.plot( prediction.numpy(), label='Prediction')#np.arange(len(temperature_data[0])),
plt.legend()
plt.show()