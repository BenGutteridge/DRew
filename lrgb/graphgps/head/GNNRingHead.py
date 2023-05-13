import torch.nn as nn

from torch_geometric.graphgym.register import register_head


class GNNRingHead(nn.Module):
    """
    GNN prediction head for ring transfer dataset task.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.layer_post_mp = nn.Linear(dim_in, dim_out)

    def forward(self, batch):
        x = batch.x[batch.mask]
        x = self.layer_post_mp(x)
        pred, label = x, batch.y
        return pred, label
    
register_head('ringtransfer', GNNRingHead)
