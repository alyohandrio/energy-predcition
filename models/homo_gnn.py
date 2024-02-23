import torch.nn as nn
from torch_geometric.nn import GCNConv, Linear
import torch_scatter

class HomoGNN(nn.Module):
    def __init__(self, conv_type, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = conv_type(-1, hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = conv_type(hidden_channels, out_channels, add_self_loops=False)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.head = Linear(out_channels, 1)
    
    def forward(self, data, group, **batch):
        x = data.x
        edge_index = data.edge_index
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        out = self.head(torch_scatter.scatter(x, group, dim=0, reduce='sum').relu())
        return out.flatten()
