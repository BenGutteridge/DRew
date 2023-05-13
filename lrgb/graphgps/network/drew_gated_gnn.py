import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvLayer
from graphgps.layer.gcnii_conv_layer import GCN2ConvLayer
from graphgps.layer.mlp_layer import MLPLayer
from graphgps.layer.drew_gatedgcn_layer import DRewGatedGCNLayer

class DRewGatedGNN(torch.nn.Module):
    """
    Custom GNN (whole network) function to replace 'custom_gnn' specifically for DRew-GatedGCN. 
    Utilises the drew_gatedgcn_layer.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner
            self.encoder = torch.nn.Sequential([self.encoder, self.pre_mp])

        assert cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        conv_model = DRewGatedGCNLayer
        self.model_type = cfg.gnn.layer_type
        layers = []
        for t in range(cfg.gnn.layers_mp):
            layers.append(conv_model(t, dim_in, dim_in,
                                     dropout=cfg.gnn.dropout,
                                     residual=cfg.gnn.residual))
        self.gnn_layers = torch.nn.ModuleList(layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)


    def forward(self, batch):
        batch = self.encoder(batch)             # Encoder (+ Pre-MP)
        xs = []
        for t in range(len(self.gnn_layers)):   # Message Passing
            xs.append(batch.x)
            batch = self.gnn_layers[t](t, xs, batch) 
        batch = self.post_mp(batch)             # (Post-MP +) Head
        return batch


register_network('drew_gated_gnn', DRewGatedGNN)
