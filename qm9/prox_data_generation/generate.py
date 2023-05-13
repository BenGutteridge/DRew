import pickle
from pathlib import Path
import os.path as osp
import torch

from find_coloring import find_coloring
from graph_modifier import modify_add_edge

import tqdm
import numpy as np
from scipy.sparse.csgraph import floyd_warshall
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_undirected
import networkx as nx


def gen_ff_graph(num_layers: int, layer_width: int):
    nb_nodes = num_layers * layer_width
    ff_graph = nx.Graph()
    ff_graph.add_nodes_from([x for x in range(num_layers * layer_width)])
    for i in range(num_layers - 1):
        ff_graph.add_edges_from(
            [
                (i * layer_width + j, (i + 1) * layer_width + k)
                for j in range(layer_width)
                for k in range(layer_width)
            ]
        )
    edge_index = to_undirected(
        torch.LongTensor(np.array(ff_graph.edges).T), num_nodes=nb_nodes
    )
    return edge_index, nb_nodes


def gen_yes_no_graph_pair(
    num_layers, width_layers, num_red, hop_threshold, count_threshold=3
):
    """
    Main function for generating a single graph from our synthetic dataset.

    :param num_layers: The number of feedforward layers in the graph
    :param width_layers: The width of every layer in the graph
    :param num_red: Number of red nodes.
    :param hop_threshold: Threshold distance for the classification.
    :param count_threshold: The threshold number of .
    :return: Returns a pair of two graphs, where the first one
             corresponds to a "yes" example (i.e. count_thresh - 1 blue nodes at most within
             the hop threshold of all red nodes`), and the second one
             corresponds to the "no" example (at least 1 red failing the condition
             (count_thresh blue within hop_thresh).
    """
    # edge_index = erdos_renyi_graph(num_nodes=num_nodes, edge_prob=edge_prob, directed=False)
    edge_index, num_nodes = gen_ff_graph(
        num_layers=num_layers, layer_width=width_layers
    )
    graph = Data(edge_index=edge_index, num_nodes=num_nodes)
    adj_matrix = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    sp_table = floyd_warshall(adj_matrix)

    yes_graph = find_coloring(
        graph,
        sp_table=sp_table,
        num_red=num_red,
        hop_threshold=hop_threshold,
        count_threshold=count_threshold,
        num_tries=200,
    )

    if not yes_graph:
        # Rejection sampling didn't generate the "yes" example
        return None, None

    no_graph = modify_add_edge(
        yes_graph, hop_threshold=hop_threshold, sp_table=sp_table,
    )  # Break this condition by recoloring a node blue

    if not no_graph:
        # There was no satisfying edge that we could add
        return None, None

    return yes_graph, no_graph


if __name__ == "__main__":
    for clf_threshold in [1, 3, 5, 8, 10]:
        graphs_list = []  # List of graphs
        for c_red in [3, 2, 1]:  # 3 Up to 3 red nodes
            for i in tqdm.tqdm(
                range(1500)
            ):  # The number of blue nodes won't be a parameter anymore
                layers = np.random.randint(15, 25)
                width = np.random.randint(3, 10)
                while True:
                    yes_g, no_g = gen_yes_no_graph_pair(
                        num_layers=layers,
                        width_layers=width,
                        hop_threshold=clf_threshold,
                        num_red=c_red,
                    )

                    if yes_g and no_g:
                        graphs_list.append(yes_g)
                        graphs_list.append(no_g)
                        break

        raw_dataset_path = osp.abspath(
            osp.join(
                osp.dirname(__file__),
                "..",
                "data",
                "Prox",
                str(clf_threshold) + "-Prox",
                "raw",
            )
        )
        # Store raw list
        Path(raw_dataset_path).mkdir(parents=True, exist_ok=True)
        with open(osp.join(raw_dataset_path, "data_list.pickle"), "wb") as f:
            pickle.dump(graphs_list, f)
