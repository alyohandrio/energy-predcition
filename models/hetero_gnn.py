import torch
from torch import nn
from torch_geometric.nn import GATConv, Linear, to_hetero
import torch_scatter


class GNN(nn.Module):
    def __init__(self, conv_type, hidden_channels, out_channels, conv_args={}):
        super().__init__()
        self.conv1 = conv_type((-1, -1), hidden_channels, **conv_args)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = conv_type((-1, -1), out_channels, **conv_args)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x

class HeteroGNN(nn.Module):
    def __init__(self, gnn_type, out_channels, head_dim, gnn_args={}, encoder_type=None, encoder_args={}, aggr='sum'):
        super().__init__()
        metadata = (['C', 'H'], [('C', 'CC', 'C'), ('C', 'CH', 'H'), ('H', 'HC', 'C'), ('H', 'HH', 'H')])
        if encoder_type is not None:
            encoder = encoder_type(out_channels=out_channels, **encoder_args)
            self.encoder = to_hetero(encoder, metadata, aggr=aggr)
        else:
            self.encoder = None
        gnn = gnn_type(out_channels=out_channels, **gnn_args)
        self.gnn = to_hetero(gnn, metadata, aggr=aggr)
        self.C_head = nn.Linear(out_channels, head_dim)
        self.H_head = nn.Linear(out_channels, head_dim)
        self.head = nn.Linear(head_dim * 2, 1)
    
    def forward(self, data, C_group, H_group, **batch):
        if self.encoder is not None:
            x_dict = self.encoder(data.x_dict, data.edge_index_dict)
        else:
            x_dict = data.x_dict
        x = self.gnn(x_dict, data.edge_index_dict)
        C_out = self.C_head(torch_scatter.scatter(x['C'], C_group, dim=0, reduce='sum').relu())
        H_out = self.H_head(torch_scatter.scatter(x['H'], H_group, dim=0, reduce='sum').relu())
        return self.head(torch.cat([C_out, H_out], dim=-1).relu()).flatten()
