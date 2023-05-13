import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.models.gnn import GNNLayer
from torch_geometric.graphgym.config import cfg
import torch_geometric.graphgym.register as register
import torch_geometric as pyg
from .drew_gnn import get_laplacian

class GNNStackStage(nn.Module):
    """
    Simple Stage that stack GNN layers

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of GNN layers
    """
    def __init__(self, dim_in, dim_out, num_layers):
        super(GNNStackStage, self).__init__()
        self.num_layers = num_layers
        for i in range(num_layers):
            if cfg.gnn.stage_type == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            else:
                d_in = dim_in if i == 0 else dim_out
            layer = GNNLayer(d_in, dim_out)
            self.add_module('layer{}'.format(i), layer)

    def forward(self, batch):
        """"""
        for i, layer in enumerate(self.children()):
            x = batch.x
            batch = layer(batch)
            if cfg.gnn.stage_type == 'stack_residual':
                batch.x = x + batch.x
        if cfg.gnn.l2norm:
            batch.x = F.normalize(batch.x, p=2, dim=-1)
        return batch
    
register.register_stage('custom_stack', GNNStackStage)
register.register_stage('stack_residual', GNNStackStage)