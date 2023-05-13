import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from typing import Callable, Union, Optional
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size, List

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset

from torch_sparse import SparseTensor, matmul
from functools import partial

class DRewGIN(nn.Module):
    """
    

    num parameters scales as d**2 * L(|E| + (L+1)/2) for hidden dim d, L layers, num edges types E
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner
        else:
            self.pre_mp = nn.Identity()

        assert cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        layers = []
        for t in range(cfg.gnn.layers_mp):
            layers.append(DRewGINLayer(t, dim_in))
        self.gnn_layers = torch.nn.ModuleList(layers)     

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        batch = self.encoder(batch)
        batch = self.pre_mp(batch)
        xs = []
        for t, layer in enumerate(self.gnn_layers):
            xs.append(batch.x) # does this change the xs element?
            batch = layer(t, xs, batch)
            # taken out of GINEConvLayer
            batch.x = F.relu(batch.x)
            batch.x = F.dropout(batch.x, p=cfg.gnn.dropout, training=self.training)
            if cfg.gnn.residual:
                batch.x = xs[-1] + batch.x  # residual connection. Essentially done already with MLP_s
        batch = self.post_mp(batch)
        return batch


register_network('drew_gin', DRewGIN)

class DRewGINLayer(nn.Module):
    """
    Just a nn.Module wrapper for the MessagePassing SPNConv
    """
    def __init__(self, t, hidden_dim):
        super().__init__()
        d = hidden_dim
        gin_nn_post = nn.Sequential(
            # pyg_nn.Linear(d, d), 
            # nn.ReLU(),
            # pyg_nn.Linear(d, d),
            )
        lin = partial(pyg_nn.Linear, d, d)
        if cfg.gnn.batchnorm:
            bn = partial(nn.BatchNorm1d, d, eps=cfg.bn.eps, momentum=cfg.bn.mom)
        else:
            bn = nn.Identity

        # alpha = torch.nn.Parameter(torch.randn(t+1))
        mlp_s = nn.Sequential(lin(), bn(), nn.ReLU()) # self-connection
        mlp_k = nn.ModuleList([nn.Sequential(pyg_nn.Linear(d, d), bn(), nn.ReLU()) for k in range(1, t+2)]) # k-hop connections
        # mlp_e = nn.ModuleList([nn.Sequential(pyg_nn.Linear(d, d), bn(), nn.ReLU()) for _ in range(len(cfg.edge_types))]) # 1-hop edge-type connections
        all_modules = dict(gin_nn_post=gin_nn_post, # making an attr of the module so it shows in model summary (hopefully)
                            # alpha=alpha,
                            mlp_s=mlp_s,
                            mlp_k=mlp_k,
                            # mlp_e=mlp_e,
                            )
        self.model = DRewGINConv(all_modules)

    def forward(self, t, xs, batch):
        batch.x = self.model(t, xs, batch.edge_index, 
                             batch.edge_attr, # for k-hop labels
                             )
        return batch


class DRewGINConv(MessagePassing):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

    or

    .. math::
        \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
        (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, modules: nn.ModuleDict,
                 eps: float = 0., train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(DRewGINConv, self).__init__(**kwargs)
        self.nu = cfg.nu if cfg.nu != -1 else float('inf')
        self.nn_post = modules['gin_nn_post']
        # self.alpha = modules['alpha'] # for the k-hop aggregations weights
        self.mlp_s = modules['mlp_s'] # for the self-connection ((1+eps) weighted)
        self.mlp_k = modules['mlp_k'] # for the k-hop aggregations (list)
        # self.mlp_e = modules['mlp_e'] # for the edge-type aggregations (list)
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn_post)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, t: int, # current layer/timestep
                    xs: List[Tensor],
                    edge_indices: List[Adj], # TODO: what is adj? a 2xN tensor?
                    edge_attr: OptTensor = None,
                    size: Size = None) -> Tensor:

        x = xs[-1]

        A = lambda k : edge_indices[:, edge_attr==k]
        mlp = lambda k : self.mlp_k[k-1] # k=1 is the first element in the list
        # alpha_weights = F.softmax(self.alpha, dim=0) # convex combination
        # alpha = lambda k : alpha_weights[k-1] # k=1 is the first element in the list

        # weighted self connection
        out = (1 + self.eps) * self.mlp_s(x)

        for k in range(1, t+2): # doesn't skip if A(k) empty - by design
            if A(k).shape[1] > 0: # shouldn't happen unless L > graph diameter
                delay = max(k - self.nu, 0)
                out += mlp(k)(self.propagate(A(k), x=xs[t-delay], size=size))

        return self.nn_post(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)
