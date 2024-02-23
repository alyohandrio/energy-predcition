import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, num_layers=1, hidden_dim=None):
        super().__init__()
        if num_layers == 1:
            self.mlp = nn.Linear(2, 1)
        elif num_layers == 2:
            self.mlp = nn.Sequential(nn.Linear(2, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, 1)
                                    )
        else:
            self.mlp = nn.Sequential(nn.Linear(2, hidden_dim),
                                     nn.ReLU(),
                                     *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(num_layers - 2)],
                                     nn.Linear(hidden_dim, 1)
                                    )
    
    def forward(self, C_group, H_group, **batch):
        C_counts = torch.unique(C_group, sorted=True, return_counts=True)[1]
        H_counts = torch.unique(H_group, sorted=True, return_counts=True)[1]
        x = torch.cat([C_counts[:,None], H_counts[:,None]], dim=1).float()
        return self.mlp(x).flatten()
