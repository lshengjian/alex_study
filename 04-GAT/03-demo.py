from torch.nn import Linear, ReLU
from torch_geometric.nn import Sequential, GATConv
from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = Planetoid(root='d:/dataset', name='Cora')
data = dataset[0].to(device)

in_channels,out_channels=dataset.num_node_features,dataset.num_classes
model = Sequential('x, edge_index', [
    (GATConv(in_channels, 64), 'x, edge_index -> x'),
    ReLU(inplace=True),
    (GATConv(64, 64), 'x, edge_index -> x'),
    ReLU(inplace=True),
    Linear(64, out_channels),
]).to(device)
#model = Model().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = F.log_softmax(model(data.x,data.edge_index))
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    print(loss.detach().cpu().item())
model.eval()
pred = model(data.x,data.edge_index).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')


