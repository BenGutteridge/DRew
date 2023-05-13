import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_stage
import torch
from .example import GNNLayer
from .stage_inits import init_SP_GCN

class SP_GNNStage(nn.Module):
    """
    For implementing SP-GCN, the nondynamic version of r*GCN
     i.e. no dynamically added components

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of GNN layers
    """
    def __init__(self, dim_in, dim_out, num_layers):
        super().__init__()
        # W_t only, not W_{k,t}
        # alpha_k sums to 1 and weights Sk
        # all Sk used at every layer - nondynamic
        alpha = min(num_layers, cfg.k_max)
        print('Running SP_GNN, alpha = ', alpha)
        self = init_SP_GCN(self, 
                                        dim_in, dim_out, 
                                        num_layers, 
                                        max_k=alpha)
        for t in range(num_layers):
            assert len(self.W[t]) == self.max_k

    def forward(self, batch):
        """
        Sum of all S_k.W_{k,t} for k < alpha
        """
        # k-hop adj matrix
        A = lambda k : batch.edge_index[:, batch.edge_attr==k]
        W = lambda t,k : self.W[t][k-1] # W(t,k)
        alpha = self.max_k
        # run through layers
        t = 0
        for t in range(self.num_layers):
            x = batch.x
            batch.x = torch.zeros_like(x)
            for k in range(1, alpha+1):
                if A(k).shape[1] > 0: # prevents adding I*W*H (bc of self added connections to zero adj)
                    batch.x = batch.x + W(t,k)(batch, x, A(k)).x
            batch.x = x + nn.ReLU()(batch.x)
            if cfg.gnn.l2norm: # normalises after every layer
                batch.x = F.normalize(batch.x, p=2, dim=-1)
        return batch

register_stage('sp_gnn', SP_GNNStage)