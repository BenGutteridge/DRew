import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_layer

import torch_geometric as pyg
from torch_geometric.graphgym.models.layer import LayerConfig

# @register_layer('drew_gcnconv')
class DRewGCNconv(nn.Module):
    """
    Graph Convolutional Network (GCN) layer

    This one allows a custom edge_index to be passed in
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.GCNConv(layer_config.dim_in, layer_config.dim_out,
                                    bias=layer_config.has_bias)

    def forward(self, batch, x, edge_index):
        batch.x = self.model(x, edge_index)
        return batch

register_layer('drew_gcnconv', DRewGCNconv)