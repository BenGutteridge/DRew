import random
from collections import defaultdict
from itertools import product
from typing import Callable, Optional

import torch
import numpy as np
from torch import Tensor

def index_to_mask(index: Tensor, size: Optional[int] = None) -> Tensor:
    r"""Converts indices to a mask representation.

    Args:
        idx (Tensor): The indices.
        size (int, optional). The size of the mask. If set to :obj:`None`, a
            minimal sized output mask is returned.

    Example:

        >>> index = torch.tensor([1, 3, 5])
        >>> index_to_mask(index)
        tensor([False,  True, False,  True, False,  True])

        >>> index_to_mask(index, size=7)
        tensor([False,  True, False,  True, False,  True, False])
    """
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask

from torch_geometric.data import Data, HeteroData, InMemoryDataset
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected

from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.graphgym.config import cfg

# from torch_geometric.graphgym.utils.ben_utils import get_k_hop_adjacencies


class RingTransferDataset(InMemoryDataset):
    r"""A synthetic dataset that returns a Ring Transfer dataset.

    Args:
        num_graphs (int, optional): The number of graphs. (default: :obj:`1`)
        num_nodes (int, optional): The average number of nodes in a graph.
            (default: :obj:`1000`)
        num_classes (int, optional): The number of node features.
            (default: :obj:`64`)
        task (str, optional): Whether to return node-level or graph-level
            labels (:obj:`"node"`, :obj:`"graph"`, :obj:`"auto"`).
            If set to :obj:`"auto"`, will return graph-level labels if
            :obj:`num_graphs > 1`, and node-level labels other-wise.
            (default: :obj:`"auto"`)
        is_undirected (bool, optional): Whether the graphs to generate are
            undirected. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        **kwargs (optional): Additional attributes and their shapes
            *e.g.* :obj:`global_features=5`.
    """
    def __init__(
        self,
        num_graphs,
        num_nodes,
        num_classes,
        # task: str = "auto",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__('.', transform)

        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self._num_classes = num_classes
        self.kwargs = kwargs
        if cfg.gnn.layers_mp == 1: # the default - otherwise use specified no.
            cfg.gnn.layers_mp = num_nodes//2
        if cfg.gnn.stage_type == 'drew_gnn':
            n_layers, d = cfg.gnn.layers_mp, cfg.gnn.dim_inner
            cfg.gnn.dim_inner = round((2 * (d ** 2) / (n_layers+1)) ** 0.5) # should set param count to roughly the same for fixed L
        split = (self.num_graphs * torch.tensor(cfg.dataset.split)).long()
        data_list, split = self.load_ring_transfer_dataset(self.num_nodes, 
                                                           split=split,
                                                           classes=self._num_classes)        
        
        self.data, self.slices = self.collate(data_list)
        
        # add train/val split masks
        self.data.train_mask = index_to_mask(torch.tensor(split[0]), size=len(self.data.x))
        self.data.val_mask = index_to_mask(torch.tensor(split[1]), size=len(self.data.x))
        self.data.test_mask = index_to_mask(torch.tensor(split[2]), size=len(self.data.x))
        

    def load_ring_transfer_dataset(self, nodes=10, split=[5000, 500, 500], classes=5):
        train = self.generate_ring_transfer_graph_dataset(nodes, classes=classes, samples=split[0])
        val = self.generate_ring_transfer_graph_dataset(nodes, classes=classes, samples=split[1])
        test = self.generate_ring_transfer_graph_dataset(nodes, classes=classes, samples=split[2])
        dataset = train + val + test
        return dataset, [list(range(int(split[i]))) for i in range(3)]

    def generate_ring_transfer_graph_dataset(self, nodes, classes=5, samples=10000):
        # Generate the dataset
        dataset = []
        samples_per_class = torch.div(samples, classes, rounding_mode="floor")
        for i in range(samples):
            label = torch.div(i, samples_per_class, rounding_mode="floor")
            target_class = np.zeros(classes)
            target_class[label] = 1.0
            graph = self.generate_ring_transfer_graph(nodes, target_class)
            dataset.append(graph)
        return dataset

    def generate_ring_transfer_graph(self, nodes, target_label):
        opposite_node = nodes // 2

        # Initialise the feature matrix with a constant feature vector
        x = np.ones((nodes, len(target_label)))

        x[0, :] = 0.0
        x[opposite_node, :] = target_label
        x = torch.tensor(x, dtype=torch.float32)

        edge_index = []
        for i in range(nodes-1):
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i])

        # Add the edges that close the ring
        edge_index.append([0, nodes - 1])
        edge_index.append([nodes - 1, 0])
        edge_index = torch.tensor(edge_index, dtype=torch.long).T

        # Create a mask for the target node of the graph
        mask = torch.zeros(nodes, dtype=torch.bool)
        mask[0] = 1

        # Add the label of the graph as a graph label
        y = torch.tensor([np.argmax(target_label)], dtype=torch.long)

        return Data(x=x, edge_index=edge_index, mask=mask, y=y)
