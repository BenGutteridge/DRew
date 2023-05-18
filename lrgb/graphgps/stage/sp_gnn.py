import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_stage
import torch
from .example import GNNLayer
from .stage_inits import init_SP_GCN

class SP_GNNStage(nn.Module):
    """
    SP-GNN: the stage for running SP-GCN

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of GNN layers
    """
    def __init__(self, dim_in, dim_out, num_layers):
        super().__init__()
        k_max = min(num_layers, cfg.k_max)
        print('Running SP_GNN, k = ', k_max)
        self = init_SP_GCN(self, 
                            dim_in, dim_out, 
                            num_layers, 
                            max_k=k_max)

    def forward(self, batch):
        # k-hop adj matrix
        A = lambda k : batch.edge_index[:, batch.edge_attr==k]
        W = lambda t,_ : self.W[t]
        alpha = lambda t,k : F.softmax(self.alpha[t], dim=0)[k-1]
        k_max = self.max_k

        t = 0
        for t in range(self.num_layers):
            x = batch.x
            batch.x = torch.zeros_like(x)
            for k in range(1, k_max+1):
                if A(k).shape[1] > 0:
                    if self.dense: batch.x = batch.x + W(t,k)(batch, x, A(k)).x
                    else: batch.x = batch.x + alpha(t,k) * W(t,k)(batch, x, A(k)).x
            batch.x = x + nn.ReLU()(batch.x)
            if cfg.gnn.l2norm:
                batch.x = F.normalize(batch.x, p=2, dim=-1)
        return batch

register_stage('sp_gnn', SP_GNNStage)
