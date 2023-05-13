import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_layer

# Note: A registered GNN layer should take 'batch' as input
# and 'batch' as output

# Example 1: Directly define a GraphGym format Conv
# take 'batch' as input and 'batch' as output
class ExampleConv1(MessagePassing):
    r"""Example GNN layer

    """
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super(ExampleConv1, self).__init__(aggr=cfg.gnn.agg, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, batch):
        """"""
        x, edge_index = batch.x, batch.edge_index
        x = torch.matmul(x, self.weight)

        batch.x = self.propagate(edge_index, x=x)

        return batch

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


# Remember to register your layer!
register_layer('exampleconv1', ExampleConv1)


# Example 2: First define a PyG format Conv layer
# Then wrap it to become GraphGym format
class ExampleConv2Layer(MessagePassing):
    r"""Example GNN layer

    """
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super(ExampleConv2Layer, self).__init__(aggr=cfg.gnn.agg, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, edge_index):
        """"""
        x = torch.matmul(x, self.weight)

        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class ExampleConv2(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(ExampleConv2, self).__init__()
        self.model = ExampleConv2Layer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


# Remember to register your layer!
register_layer('exampleconv2', ExampleConv2)
