import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_scatter import scatter

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_layer

from param_calcs import get_k_neighbourhoods


class DRewGatedGCNLayer(pyg_nn.conv.MessagePassing):
    """
        GatedGCN layer
        Residual Gated Graph ConvNets
        https://arxiv.org/pdf/1711.07553.pdf
    """
    def __init__(self, t, in_dim, out_dim, dropout, residual,
                 equivstable_pe=False, **kwargs):
        super().__init__(**kwargs)
        self.share_weights = bool('share' in cfg.gnn.layer_type)
        self.A = pyg_nn.Linear(in_dim, out_dim, bias=True)
        # self.C = pyg_nn.Linear(in_dim, out_dim, bias=True) # leave for now
        self.D = pyg_nn.Linear(in_dim, out_dim, bias=True)
        if self.share_weights:
            self.B = pyg_nn.Linear(in_dim, out_dim, bias=True)
            self.E = pyg_nn.Linear(in_dim, out_dim, bias=True)
        else:
            k_neighbourhoods = get_k_neighbourhoods(t)
            self.B = nn.ModuleDict({str(k): pyg_nn.Linear(in_dim, out_dim, bias=True) for k in k_neighbourhoods})
            self.E = nn.ModuleDict({str(k): pyg_nn.Linear(in_dim, out_dim, bias=True) for k in k_neighbourhoods})


        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        self.EquivStablePE = equivstable_pe
        if self.EquivStablePE:
            self.mlp_r_ij = nn.Sequential(
                nn.Linear(1, out_dim), nn.ReLU(),
                nn.Linear(out_dim, 1),
                nn.Sigmoid())

        self.bn_node_x = nn.BatchNorm1d(out_dim)
        self.bn_edge_e = nn.BatchNorm1d(out_dim)
        self.dropout = dropout
        self.residual = residual
        self.e = None

        self.nu = cfg.nu if cfg.nu != -1 else float('inf')

    def forward(self, t, xs, batch): # needs to take current layer and custom x list
        x, edge_index = batch.x, batch.edge_index
        # e = batch.edge_attr
        """
        x               : [n_nodes, in_dim]
        e               : [n_edges, in_dim]     # not currently used
        edge_index      : [2, n_edges]
        """

        if self.residual:
            x_in = x
            # e_in = e

        k_neighbourhoods = get_k_neighbourhoods(t)
        delay = lambda k : max(k-self.nu, 0)
        if self.share_weights:
            B = lambda _ : self.B
            E = lambda _ : self.E
        else:
            B = lambda k : self.B[str(k)]
            E = lambda k : self.E[str(k)]

        Bx = lambda k: B(k)(xs[t-delay(k)])
        Ex = lambda k: E(k)(xs[t-delay(k)])
        
        Ax = self.A(x)
        # Ce = self.C(e)x
        Dx = self.D(x) # these use the local node i and do not require varying k-neighbourhoods

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        pe_LapPE = batch.pe_EquivStableLapPE if self.EquivStablePE else None

        # make Dx_i, Ex_j, Bx_j
        i_idxs, j_idxs = edge_index[1,:], edge_index[0,:] # 0 is j, 1 is i by pytorch's convention
        node_dim = 0 # default is -2 which is equivalent
        Dx_i = lambda k : Dx.index_select(node_dim, i_idxs[batch.edge_attr==k])
        Ex_j = lambda k : Ex(k).index_select(node_dim, j_idxs[batch.edge_attr==k]) # D,E for calculating edge gates e_ij
        Bx_j = lambda k : Bx(k).index_select(node_dim, j_idxs[batch.edge_attr==k]) # B for weighting edge gates
        
        if pe_LapPE: # TODO
            PE_i = pe_LapPE.index_select(node_dim, i_idxs)
            PE_j = pe_LapPE.index_select(node_dim, i_idxs)

        # MESSAGES
        sigma_ij = lambda k : torch.sigmoid(Dx_i(k) + Ex_j(k)) # \sigma(e_ij)

        # # Handling for Equivariant and Stable PE using LapPE
        # # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        if self.EquivStablePE:
            r_ij = ((PE_i - PE_j) ** 2).sum(dim=-1, keepdim=True)
            r_ij = self.mlp_r_ij(r_ij)  # the MLP is 1 dim --> hidden_dim --> 1 dim
            sigma_ij = lambda k : torch.sigmoid(Dx_i(k) + Ex_j(k)) * r_ij

        # self.e = e_ij
        
        # AGGREGATE
        dim_size = self.get_dim_size(xs)
        alpha = torch.ones(len(k_neighbourhoods))
        alpha = F.softmax(alpha, dim=0)
        if not cfg.agg_weights.convex_combo: alpha = alpha * alpha.shape[0]
        x = 0
        for k_i, k in enumerate(k_neighbourhoods):
            sum_sigma_x = sigma_ij(k) * Bx_j(k)
            numerator_eta_xj = scatter(sum_sigma_x, i_idxs[batch.edge_attr==k], 
                                    0, None, dim_size,
                                    reduce='sum')
            sum_sigma = sigma_ij(k)
            denominator_eta_xj = scatter(sum_sigma, i_idxs[batch.edge_attr==k],
                                        0, None, dim_size,
                                        reduce='sum')
            eta_xj = (numerator_eta_xj / (denominator_eta_xj + 1e-6))
            x = x + alpha[k_i] * eta_xj

        # # UPDATE
        x = Ax + x

        x = self.bn_node_x(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        # e = e_ij
        # e = self.bn_edge_e(e)
        # e = F.relu(e)
        # e = F.dropout(e, self.dropout, training=self.training)
        if self.residual:
            x = x_in + x
            # e = e_in + e
        batch.x = x
        # batch.edge_attr = e

        return batch

    def get_dim_size(self, xs):
        try: 
            return xs[0].shape[0]
        except: # when doing alpha
            for x_t in xs:
                if x_t is not None: 
                    return x_t.shape[0]


class DRewGatedGCNGraphGymLayer(nn.Module):
    """GatedGCN layer.
    Residual Gated Graph ConvNets
    https://arxiv.org/pdf/1711.07553.pdf
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = DRewGatedGCNLayer(in_dim=layer_config.dim_in,
                                   out_dim=layer_config.dim_out,
                                   dropout=0.,  # Dropout is handled by GraphGym's `GeneralLayer` wrapper
                                   residual=False,  # Residual connections are handled by GraphGym's `GNNStackStage` wrapper
                                   **kwargs)

    def forward(self, batch):
        return self.model(batch)


register_layer('drewgatedgcnconv', DRewGatedGCNGraphGymLayer)
register_layer('share_drewgatedgcnconv', DRewGatedGCNGraphGymLayer)
