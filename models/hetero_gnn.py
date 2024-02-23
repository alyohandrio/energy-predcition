import torch
from torch import nn
from torch_geometric.nn import GATConv, Linear, to_hetero
import torch_scatter





class GAT(nn.Module):
    def __init__(self, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, heads=heads, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, heads=heads, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x

class HeteroGNN(nn.Module):
    def __init__(self, conv_type, hidden_channels, out_channels, head_dim, heads, aggr='sum'):
        super().__init__()
        gat = conv_type(hidden_channels=hidden_channels, out_channels=out_channels, heads=heads)
        metadata = (['C', 'H'], [('C', 'CC', 'C'), ('C', 'CH', 'H'), ('H', 'HC', 'C'), ('H', 'HH', 'H')])
        self.gat = to_hetero(gat, metadata, aggr=aggr)
        self.C_head = nn.Linear(out_channels, head_dim)
        self.H_head = nn.Linear(out_channels, head_dim)
        self.head = nn.Linear(head_dim * 2, 1)
    
    def forward(self, data, C_group, H_group, **batch):
        x = self.gat(data.x_dict, data.edge_index_dict)
        C_out = self.C_head(torch_scatter.scatter(x['C'], C_group, dim=0, reduce='sum').relu())
        H_out = self.H_head(torch_scatter.scatter(x['H'], H_group, dim=0, reduce='sum').relu())
        return self.head(torch.cat([C_out, H_out], dim=-1).relu()).flatten()
