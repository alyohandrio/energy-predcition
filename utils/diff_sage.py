import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn

from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear


class DiffSAGEConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        aggr = "mean",
        transform = None,
        normalize = False,
        bias = True,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.transform = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if aggr == 'lstm':
            kwargs.setdefault('aggr_kwargs', {})
            kwargs['aggr_kwargs'].setdefault('in_channels', in_channels[0])
            kwargs['aggr_kwargs'].setdefault('out_channels', in_channels[0])

        super().__init__(aggr, **kwargs)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                in_channels[0])
        else:
            aggr_out_channels = in_channels[0]

        self.lin_l = Linear(aggr_out_channels, out_channels, bias=bias)
        self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(
        self,
        x,
        edge_index,
        size = None,
    ):

        if isinstance(x, Tensor):
            x = (x, x)

        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_i, x_j):
        if self.transform is not None:
            return self.transform(x_j - x_i)
        return x_j - x_i


class DiffSAGEWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.sage = DiffSAGEConv(*args, **kwargs)

    def forward(self, x, edge_index, size=None):
        return self.sage(x, edge_index, size)
