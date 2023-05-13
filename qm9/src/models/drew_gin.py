import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, Embedding
from torch_scatter import scatter_max, scatter_mean, scatter_sum
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from models.layers.hsp_gin_layer import instantiate_mlp
from models.layers import Share_DRew_GIN_Layer, DRew_GIN_Layer

from .hsp_gin import get_laplacian
import numpy as np

# Modes: GC: Graph Classification.
GRAPH_CLASS = "gc"
GRAPH_REG = "gr"


class Net_DRew_GIN(torch.nn.Module):
    def __init__(
        self,
        nu,
        num_features,
        num_classes,
        emb_sizes=None,
        max_distance=5,
        mode=GRAPH_CLASS,
        ogb_gc=None,
        eps=0,
        drpt_prob=0.5,
        scatter="max",
        inside_aggr="sum",
        outside_aggr="weight",
        device="cpu",
        batch_norm=True,
        layer_norm=False,
        pool_gc=False,
        residual_frequency=-1,
        dataset=None,
        learnable_emb=False,
        use_feat=False,
        use_weight_share=False,
    ):
        super(Net_DRew_GIN, self).__init__()
        if emb_sizes is None:  # Python default handling for mutable input
            emb_sizes = [64, 64, 64]  # The 0th entry is the input feature size.
        self.num_features = num_features
        self.max_distance = max_distance
        self.emb_sizes = emb_sizes
        self.num_layers = len(self.emb_sizes) - 1
        self.eps = eps
        self.drpt_prob = drpt_prob
        self.scatter = scatter
        self.device = device
        self.mode = mode
        self.dataset = dataset

        self.inside_aggr = inside_aggr
        self.outside_aggr = outside_aggr
        self.ogb_gc = ogb_gc
        self.use_feat = use_feat  # The OGB feature use
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.pool_gc = pool_gc
        self.residual_freq = residual_frequency
        self.learnable_emb = learnable_emb

        if use_weight_share:
            delay_layer = Share_DRew_GIN_Layer
        else:
            delay_layer = DRew_GIN_Layer

        additional_kwargs = {"edgesum_relu": True}
        if self.ogb_gc is not None:  # This needs dedicated encoders
            if self.ogb_gc in [
                "ogbg-molhiv",
                "ogbg-molpcba",
                "ogbg-molbbbp",
                "ogbg-molbace",
                "ogbg-molclintox",
                "ogbg-moltox21",
                "ogbg-moltoxcast",
                "ogbg-molsider",
                "ogbg-molmuv",
            ]:
                self.encoder = AtomEncoder(
                    emb_dim=emb_sizes[0]
                )  # This supersedes the initial MLP
                self.edge_encoders = torch.nn.ModuleList(
                    [BondEncoder(emb_dim=emb_sizes[0]) for i in range(self.num_layers)]
                )

            elif self.ogb_gc == "ogbg-ppa":
                self.encoder = torch.nn.Embedding(1, emb_sizes[0])
                self.edge_encoder = torch.nn.Linear(
                    7, emb_sizes[0]
                )
        elif self.dataset and self.dataset.endswith("Prox"):
            self.initial_embedding = Embedding(10, self.emb_sizes[0]).requires_grad_(
                self.learnable_emb
            )
            # 10 is the number of possible node colors
        elif self.dataset and self.dataset in [
            "MUV",
            "HIV",
            "BACE",
            "BBBP",
            "Tox21",
            "ToxCast",
            "SIDER",
            "ClinTox",
        ]:
            additional_kwargs["edgesum_relu"] = False
            self.atom_embedding = Embedding(120, self.emb_sizes[0])
            self.chirality_embedding = Embedding(3, self.emb_sizes[0])
        else: # APPLIES FOR QM9 -- a linear layer w batchnorm
            self.initial_mlp = instantiate_mlp(
                in_channels=num_features,
                out_channels=emb_sizes[0],
                device=device,
                batch_norm=batch_norm,
                final_activation=True,
            )
        if self.mode == GRAPH_REG:  # Mimicking the GNN-FiLM paper:
            # https://github.com/microsoft/tf-gnn-samples/blob/73e2c950736ac7f662fa88c03c9c0c45fe29d65f/tasks/qm9_task.py
            # Lines 163 - 188
            self.regression_gate_mlp = instantiate_mlp(
                in_channels=emb_sizes[-1] + num_features,
                out_channels=1,
                device=device,
                final_activation=False,
                batch_norm=batch_norm,
            )
            self.regression_transform_mlp = instantiate_mlp(
                in_channels=emb_sizes[-1],
                out_channels=1,
                device=device,
                final_activation=False,
                batch_norm=batch_norm,
            )  # No final act
        self.initial_linear = Linear(emb_sizes[0], num_classes).to(device)

        hsp_layers = []
        linears = []
        if self.layer_norm:
            layer_norms = []
        for t in range(self.num_layers):
            hsp_layer = delay_layer(
                t=t,
                nu=nu,
                in_channels=emb_sizes[t],
                out_channels=emb_sizes[t + 1],
                eps=self.eps,
                max_distance=self.max_distance,
                inside_aggr=inside_aggr,
                batch_norm=batch_norm,
                outside_aggr=outside_aggr,
                dataset=dataset,
                device=device,
                **additional_kwargs
            ).to(device)
            hsp_layers.append(hsp_layer)
            if self.layer_norm:
                layer_norms.append(torch.nn.LayerNorm(emb_sizes[t + 1]))
            linears.append(Linear(emb_sizes[t + 1], num_classes).to(device))

        self.hsp_modules = ModuleList(hsp_layers)
        self.linear_modules = ModuleList(linears)
        if self.layer_norm:
            self.layer_norms = ModuleList(layer_norms)

    def reset_parameters(self):
        if self.layer_norm:
            for x in self.layer_norms:
                x.reset_parameters()
        if self.ogb_gc is not None:
            if self.ogb_gc in [
                "ogbg-molhiv",
                "ogbg-molpcba",
                "ogbg-molbbbp",
                "ogbg-molbace",
                "ogbg-molclintox",
                "ogbg-moltox21",
                "ogbg-moltoxcast",
                "ogbg-molsider",
                "ogbg-molmuv",
            ]:
                for emb in self.encoder.atom_embedding_list:
                    torch.nn.init.xavier_uniform_(emb.weight.data)
                for e_enc in self.edge_encoders:
                    for emb in e_enc.bond_embedding_list:
                        torch.nn.init.xavier_uniform_(emb.weight.data)
            elif self.ogb_gc == "ogbg-ppa":
                self.encoder.reset_parameters()
                self.edge_encoder.reset_parameters()
        elif hasattr(self, "initial_mlp"):
            for module in self.initial_mlp:
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
        for (name, module) in self._modules.items():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        for module in self.hsp_modules:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        for module in self.linear_modules:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        if self.mode == GRAPH_REG:
            for module in self.regression_transform_mlp:
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            for module in self.regression_gate_mlp:
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()

    def pooling(self, x_feat, batch):
        if self.scatter == "max":
            return scatter_max(x_feat, batch, dim=0)[0].to(self.device)
        elif self.scatter == "mean":
            return scatter_mean(x_feat, batch, dim=0).to(self.device)
        else:
            pass

    def forward(self, data):
        x_feat = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_weights = data.edge_weights.to(self.device)

        if self.inside_aggr[0] == "r":  # 'Relational' variant (for QM9)
            edge_attr = data.edge_attr.to(self.device)
        else:
            edge_attr = None

        if self.mode == GRAPH_CLASS or self.mode == GRAPH_REG:
            batch = data.batch.to(self.device)

        # Input encoding
        if self.ogb_gc:
            if self.use_feat:
                x_feat = self.encoder(
                    x_feat
                )  # Use the standard OGB encoding techniques
            else:
                x_feat = self.encoder(x_feat[:, :2])  # Only use the first two features
        elif self.dataset and self.dataset.endswith("Prox"):
            x_feat = self.initial_embedding(x_feat).squeeze(1)
        elif self.dataset and self.dataset in [
            "MUV",
            "HIV",
            "BACE",
            "BBBP",
            "Tox21",
            "ToxCast",
            "SIDER",
            "ClinTox",
        ]:
            x_feat = self.atom_embedding(x_feat[:, 0]) + self.chirality_embedding(
                x_feat[:, 1]
            )
        else:
            x_feat = self.initial_mlp(x_feat)

        if self.mode == GRAPH_CLASS or (
            self.mode == GRAPH_REG and self.pool_gc
        ):  # Override option now added
            out = F.dropout(
                self.pooling(self.initial_linear(x_feat), batch), p=self.drpt_prob
            )
        elif self.mode == GRAPH_REG:
            pass  # Do nothing

        if self.residual_freq > 0:
            last_state_list = [x_feat]  # If skip connections are being used
        
        # xs = torch.zeros((len(self.hsp_modules), x_feat.shape[0], x_feat.shape[1]))#.to(self.device)
        xs = []
        for t, value in enumerate(zip(self.hsp_modules, self.linear_modules)): # ******* PER-LAYER LOOP *****************
            # xs[t] = x_feat
            xs.append(x_feat)
            hsp_layer, linear_layer = value
            if (
                self.inside_aggr == "edgesum"
            ):  # For OGBG (Only load direct edges for memory footprint reduction)
                if self.ogb_gc is not None:
                    if self.dataset == "ogbg-ppa":
                        enc = self.edge_encoder
                    else:
                        enc = self.edge_encoders[t]

                    if self.use_feat:
                        edge_embeddings = enc(
                            data.edge_attr[data.edge_weights == 1].to(self.device)
                        ).to(self.device)
                    else:
                        edge_embeddings = enc(
                            data.edge_attr[data.edge_weights == 1, :2].to(self.device)
                        ).to(self.device)

                else:
                    edge_embeddings = None
                    edge_attr = data.edge_attr.to(self.device)
            else:
                edge_embeddings = None
            x_feat = hsp_layer(
                t,
                node_embeddings=xs, # node embeddings contains all x_feats from 0...t
                edge_index=edge_index,
                edge_weights=edge_weights,
                batch=batch,
                edge_attr=edge_attr,
                direct_edge_embs=edge_embeddings,
            ).to(self.device)
            if self.residual_freq > 0:
                if self.residual_freq <= t + 1:
                    x_feat = (
                        x_feat + last_state_list[-self.residual_freq]
                    )  # Residual connection
                last_state_list.append(
                    x_feat
                )  # Add the new state to the list for easy referencing

            if self.mode == GRAPH_CLASS or (self.mode == GRAPH_REG and self.pool_gc):
                if not self.layer_norm:
                    out += F.dropout(
                        linear_layer(self.pooling(x_feat, batch)),
                        p=self.drpt_prob,
                        training=self.training,
                    )
                else:
                    out += F.dropout(
                        linear_layer(
                            self.layer_norms[t](self.pooling(x_feat, batch))
                        ),
                        p=self.drpt_prob,
                        training=self.training,
                    )
            elif self.mode == GRAPH_REG:
                if self.layer_norm:
                    x_feat = torch.relu(
                        self.layer_norms[t](x_feat)
                    )  # Just apply layer norms then. ReLU is crucial.
                    # Otherwise Layer Norm freezes
                else:
                    x_feat = torch.relu(x_feat)

        if self.mode == GRAPH_CLASS or (self.mode == GRAPH_REG and self.pool_gc):
            return out
        elif (
            self.mode == GRAPH_REG
        ):  # New OUTPUT. Concatenate input features and final reps, and use these for pooling
            gate_input = torch.cat([data.x, x_feat], dim=-1)
            gate_out = self.regression_gate_mlp(gate_input)
            transform_out = self.regression_transform_mlp(x_feat)
            product = torch.sigmoid(gate_out) * transform_out
            out = scatter_sum(product, batch, dim=0).to(self.device)
            return out

    def log_hop_weights(self, neptune_client, exp_dir):
        if self.outside_aggr in ["weight"]:
            for i, layer in enumerate(self.hsp_modules):
                data = layer.hop_coef.data
                soft_data = F.softmax(data, dim=0)
                for d, (v, sv) in enumerate(zip(data, soft_data), 1):
                    log_dir = exp_dir + "/conv_" + str(i) + "/" + "weight_" + str(d)
                    neptune_client[log_dir].log(v)
                    soft_log_dir = (
                        exp_dir + "/conv_" + str(i) + "/" + "soft_weight_" + str(d)
                    )
                    neptune_client[soft_log_dir].log(sv)
