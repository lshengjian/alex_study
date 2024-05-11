import torch
from torch_geometric.data import Data

# edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
# data = Data(x=x, edge_index=edge_index)
# print(data)

edge_index = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
data = Data(x=x, edge_index=edge_index.t().contiguous())
print(f'num_nodes:{data.num_nodes}')
print(f'num_edges:{data.num_edges}')
print(f'num_node_features:{data.num_node_features}')
assert not data.has_isolated_nodes()

from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='d:/dataset', name='Cora')
assert dataset.num_classes==7
assert dataset.num_node_features==1433

from torch_geometric.loader import DataLoader

loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    print(batch)
    break
