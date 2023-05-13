import torch
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean
from torch.nn import ModuleList, BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv


# Modes: GC: Graph Classification.
GRAPH_CLASS = "gc"


class NetGCN(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        emb_sizes=None,
        mode=GRAPH_CLASS,
        drpt_prob=0.5,
        scatter="max",
        device="cpu",
        emb_input=-1,
    ):
        super(NetGCN, self).__init__()
        if emb_sizes is None:  # Python default handling for mutable input
            emb_sizes = [32, 64, 64]  # The 0th entry is the input feature size.
        self.num_features = num_features
        self.emb_sizes = emb_sizes
        self.num_layers = len(self.emb_sizes) - 1
        self.drpt_prob = drpt_prob
        self.scatter = scatter
        self.device = device
        self.mode = mode
        self.emb_input = emb_input

        if self.emb_input > 0:
            self.initial_embedding = (
                torch.nn.Embedding(10, self.emb_sizes[0])
                .requires_grad_(True)
                .to(device)
            )

        self.initial_mlp_modules = ModuleList(
            [
                Linear(num_features, emb_sizes[0]).to(device),
                BatchNorm1d(emb_sizes[0]).to(device),
                ReLU().to(device),
                Linear(emb_sizes[0], emb_sizes[0]).to(device),
                BatchNorm1d(emb_sizes[0]).to(device),
                ReLU().to(device),
            ]
        )
        self.initial_mlp = Sequential(*self.initial_mlp_modules).to(device)
        self.initial_linear = Linear(emb_sizes[0], num_classes).to(device)

        gcn_layers = []
        linears = []
        for i in range(self.num_layers):
            in_channel = emb_sizes[i]
            out_channel = emb_sizes[i + 1]
            gcn_layer = GCNConv(in_channels=in_channel, out_channels=out_channel).to(
                device
            )
            gcn_layers.append(gcn_layer)
            linears.append(Linear(emb_sizes[i + 1], num_classes).to(device))

        self.gcn_modules = ModuleList(gcn_layers)
        self.linear_modules = ModuleList(linears)

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        for module in self.gcn_modules:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        for module in self.linear_modules:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        for module in self.initial_mlp_modules:
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

        if self.emb_input > 0:
            x_feat = self.initial_embedding(x_feat).squeeze(
                1
            )  # Pass it by the embedding
        else:
            x_feat = self.initial_mlp(x_feat)  # Otherwise by an MLP
        if self.mode == GRAPH_CLASS:
            out = F.dropout(
                self.pooling(self.initial_linear(x_feat), data.batch), p=self.drpt_prob
            )

        for gcn_layer, linear_layer in zip(self.gcn_modules, self.linear_modules):
            edges = edge_index.T[edge_weights == 1].T
            x_feat = gcn_layer(x_feat, edges).to(self.device)

            if self.mode == GRAPH_CLASS:
                out += F.dropout(
                    linear_layer(self.pooling(x_feat, data.batch)),
                    p=self.drpt_prob,
                    training=self.training,
                )

        if self.mode == GRAPH_CLASS:
            return out

    def log_hop_weights(self, neptune_client, exp_dir):
        # This is a function intended for the SPN, to keep track of weights.
        # For standard GNNs, it is of no use, so we define it as a blank fct
        pass
