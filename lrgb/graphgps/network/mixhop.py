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
from torch import nn
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset

from torch_sparse import SparseTensor, matmul
from functools import partial

from torch_geometric.utils import to_dense_adj, dense_to_sparse
from scipy import sparse
from .mixhop_layers import SparseNGCNLayer, DenseNGCNLayer, ListModule
from torch_geometric.nn.conv.gcn_conv import gcn_norm as get_normalized_adj

class MixHopGCN(nn.Module):
    """
    
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()

        if cfg.mixhop_args.max_P != 0:
            cfg.mixhop_args.layers = [cfg.gnn.dim_inner] * cfg.mixhop_args.max_P
        else:
            print('Warning: P parameter not set for MixHop; node encoder and MixHop layer dimensionalities will be inconsistent.')

        assert len(set(cfg.mixhop_args.layers)) == 1 # first layer has same dimensionality for each adj power
        dim_in = cfg.mixhop_args.layers[0]
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

        feature_number = dim_in # node feature dim
        class_number = dim_out # maybe? or whatever the output is

        self.args = cfg.mixhop_args
        self.args.dropout = cfg.gnn.dropout
        self.feature_number = feature_number 
        self.class_number = class_number
        self.setup_layer_structure() 
          
        save_num_post_mp_layers = cfg.gnn.layers_post_mp
        if cfg.gnn.layers_post_mp > 1: # if there are multiple post-mp layers we want them to have inner dim d, not P*d
            self.pre_post_mp = nn.Sequential(nn.Linear(cfg.gnn.dim_inner*cfg.mixhop_args.max_P, 
                                                       cfg.gnn.dim_inner, bias=True),
                                                       nn.ReLU())
            
            cfg.gnn.layers_post_mp -= 1
            head_dim_in = cfg.mixhop_args.layers[0]
        else: head_dim_in = sum(cfg.mixhop_args.layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        from graphgps.drew_utils import get_task_id
        if get_task_id() == 'pcqm': 
            kwargs = dict(mixhop_dims=(sum(cfg.mixhop_args.layers), cfg.mixhop_args.layers[0])) # want final layer in InductiveEdgeHead to be Pd * d, not Pd*Pd
        else: kwargs = {}
        self.post_mp = GNNHead(dim_in=head_dim_in, dim_out=dim_out, **kwargs)
        cfg.gnn.layers_post_mp = save_num_post_mp_layers

    def forward(self, batch):

        """
        """

        batch = self.encoder(batch)
        batch = self.pre_mp(batch)

        # features = batch.x
        A_tilde_hat = get_normalized_adj(batch.edge_index, add_self_loops=True)
        A_tilde_hat = dict(indices=A_tilde_hat[0],
                                           values=A_tilde_hat[1])

        for t in range(cfg.gnn.layers_mp):
            batch.x = torch.cat([self.layers[t][i](A_tilde_hat, batch.x) for i in range(len(self.layers[t]))], dim=1)
            if cfg.gnn.batchnorm: batch.x = self.batchnorm[t](batch.x)
            batch.x = nn.ReLU()(batch.x)

        if cfg.gnn.layers_post_mp > 1: batch.x = self.pre_post_mp(batch.x)
        batch = self.post_mp(batch)
        return batch

    
    # def get_normalized_adj(self, edge_index):
    #     A = to_dense_adj(edge_index).squeeze()
    #     A_tilde = A + torch.eye(A.shape[0])
    #     D = torch.diag(A_tilde.sum(axis=0)) # diagonalise row (or col) sums of adj
    #     D = D.power(-0.5)
    #     A_tilde_hat = D.dot(A_tilde).dot(D)
    #     return A_tilde_hat


    # def calculate_layer_sizes(self):
    #     self.abstract_feature_number_1 = sum(self.args.layers)
    #     # self.abstract_feature_number_2 = sum(self.args.layers_2)
    #     self.order_1 = len(self.args.layers_1)
    #     self.order_2 = len(self.args.layers_2)

    def setup_layer_structure(self):
        """
        Creating the layer structure (3 convolutional upper layers, 3 bottom layers) and dense final.
        """
        # nb replaced sparse w dense
        # self.upper_layers = [DenseNGCNLayer(self.feature_number, self.args.layers_1[i-1], i, self.args.dropout) for i in range(1, self.order_1+1)]
        # self.upper_layers = ListModule(*self.upper_layers)
        # self.bottom_layers = [DenseNGCNLayer(self.abstract_feature_number_1, self.args.layers_2[i-1], i, self.args.dropout) for i in range(1, self.order_2+1)]
        # self.bottom_layers = ListModule(*self.bottom_layers)

        if cfg.gnn.batchnorm: 
            bn = partial(nn.BatchNorm1d, sum(self.args.layers), eps=cfg.bn.eps, momentum=cfg.bn.mom)
            self.batchnorm = [bn()]
        first_layer = [DenseNGCNLayer(self.feature_number, 
                                      self.args.layers[i-1], 
                                      i, self.args.dropout) for i in range(1, len(self.args.layers)+1)]
        self.layers = [ListModule(*first_layer)]
        
        for t in range(1, cfg.gnn.layers_mp):
            self.layers.append(ListModule(*[DenseNGCNLayer(sum(self.args.layers), 
                                                          self.args.layers[i-1],
                                                          i, self.args.dropout) for i in range(1, len(self.args.layers)+1)]))
            if cfg.gnn.batchnorm: self.batchnorm.append(bn())
        self.layers = nn.ModuleList(self.layers)
        if cfg.gnn.batchnorm: self.batchnorm = nn.ModuleList(self.batchnorm)

        # self.fully_connected = torch.nn.Linear(self.abstract_feature_number_2, self.class_number)


register_network('mixhop_gcn', MixHopGCN)