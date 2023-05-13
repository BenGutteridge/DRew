import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_stage
import torch
from .example import GNNLayer
# from .utils import init_khop_GCN
from .stage_inits import init_DRewGCN, init_shareDRewGCN
sort_and_removes_dupes = lambda mylist : sorted(list(dict.fromkeys(mylist)))
from param_calcs import get_k_neighbourhoods

# @register_stage('drew_gnn')      # xt+1 = f(x)       (NON-RESIDUAL)
class DRewShareGNNStage(nn.Module):
    """
    DRewGNN stage with weight sharing

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of GNN layers
    """
    def __init__(self, dim_in, dim_out, num_layers):
        super().__init__()
        self = init_shareDRewGCN(self, dim_in, dim_out, num_layers)

    def forward(self, batch):
        """
        x_{t+1} = x_t + f(x_t, x_{t-1})
        first pass: uses regular edge index for each layer
        """

        # new k-hop method: efficient
        # k-hop adj matrix
        A = lambda k : batch.edge_index[:, batch.edge_attr==k]
        W = lambda t : self.W_t["t=%d"%(t)]

        # run through layers
        t, x = 0, [] # length t list with x_0, x_1, ..., x_t
        # modules = self.children()
        for t in range(self.num_layers):
            x.append(batch.x)
            batch.x = torch.zeros_like(x[t])
            k_neighbourhoods = get_k_neighbourhoods(t)
            alpha = self.alpha_t[t] if cfg.agg_weights.use else torch.ones(len(k_neighbourhoods)) # learned weighting or equal weighting
            alpha = F.softmax(alpha, dim=0)
            alpha = alpha if cfg.agg_weights.convex_combo else alpha * len(k_neighbourhoods) # convex comb, or scale by no. of terms (e.g. unity weights for agg_weights.use=False)
            for i, k in enumerate(k_neighbourhoods):
                if A(k).shape[1] > 0: # iff there are edges of type k
                    delay = max(k-self.nu,0)
                    batch.x = batch.x + alpha[i] * W(t)(batch, x[t-delay], A(k)).x
            batch.x = x[t] + nn.ReLU()(batch.x)
            if cfg.gnn.l2norm: # normalises after every layer
                batch.x = F.normalize(batch.x, p=2, dim=-1)
        return batch

register_stage('drew_share_gnn', DRewShareGNNStage)

import numpy as np

def get_laplacian(edge_index):
    L = pyg.utils.get_laplacian(edge_index, normalization='sym')[0]
    L = pyg.utils.to_dense_adj(L).squeeze() # from index format to matrix
    return tonp(L)

def tonp(tsr):
    if isinstance(tsr, np.ndarray):
        return tsr
    elif isinstance(tsr, np.matrix):
        return np.array(tsr)
    # elif isinstance(tsr, scipy.sparse.csc.csc_matrix):
    #     return np.array(tsr.todense())

    assert isinstance(tsr, torch.Tensor)
    tsr = tsr.cpu()
    assert isinstance(tsr, torch.Tensor)

    try:
        arr = tsr.numpy()
    except TypeError:
        arr = tsr.detach().to_dense().numpy()
    except:
        arr = tsr.detach().numpy()

    assert isinstance(arr, np.ndarray)
    return arr