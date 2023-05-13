import torch
import numpy as np
from torch_geometric.data import Data


def modify_recolor(yes_graph, hop_threshold, sp_table, count_threshold=3):
    """
    Makes a new node blue, to break the rule (no new edges)

    :param yes_graph: A graph that doesn't have a RB pair within a distance of ``hop_threshold``.
    :param hop_threshold: The classification threshold for the maximum distance.
    :param sp_table: A table with dimensions ``(N, N)`` that contains the shortest paths between each pair of nodes.
    :param count_threshold: The threshold number of blue nodes not supposed to appear within hop_thresh from red.

    :return: The desired "no" instance for the dataset.
    """
    # Similar logic to initial color generation
    node_colors = np.copy(yes_graph.x.cpu().numpy())[:, 0]
    red_nodes = np.arange(yes_graph.num_nodes)[node_colors == 1]  # Red Nodes
    blue_nodes = np.arange(yes_graph.num_nodes)[node_colors == 2]  # Blue Nodes
    num_red = red_nodes.shape[0]
    within_thresh_mat = np.zeros((yes_graph.num_nodes, num_red))  # [N, #Red]
    # Sequential generation
    for idx, red_node in enumerate(red_nodes):
        within_thresh_nodes = np.arange(yes_graph.num_nodes)[
            sp_table[red_node, :] <= hop_threshold
        ]  # Get possible Bl
        within_thresh_mat[within_thresh_nodes, idx] = 1
    within_thresh_mat[red_nodes, :] = np.full(
        (1, num_red), count_threshold
    )  # To avoid sampling a red node
    within_thresh_mat[blue_nodes, :] = np.full(
        (1, num_red), count_threshold
    )  # To avoid sampling a blue node
    # (must be last)
    # We now have a matrix [N, #Red] saying which node is within hop_thresh of each red. We now sample from these
    running_thresh = np.full(
        (1, num_red), count_threshold - 1, dtype=np.float64
    )  # We're continuing from count - 1
    target = np.full_like(
        running_thresh, count_threshold
    )  # We want to have at least one count by the end
    can_finish = True
    while np.min(target - running_thresh) > 0:  # While no nodes work
        gap = target - running_thresh
        valid_completions = np.logical_and(
            np.all(within_thresh_mat <= gap, axis=1),
            np.sum(1 * within_thresh_mat, axis=1) > 0,
        )  # Pick "valid" nodes
        nb_valid_completions = np.sum(1 * valid_completions)
        if nb_valid_completions == 0:
            can_finish = False
            break
        else:  # There is a valid sample
            indices = np.arange(yes_graph.num_nodes)[
                valid_completions
            ]  # Get the potential candidates
            blue_node = np.random.choice(
                indices, size=1, replace=False
            )  # TODO: Can make smarter
            running_thresh += within_thresh_mat[
                blue_node, :
            ]  # Add to the running counter
            within_thresh_mat[
                blue_node
            ] = count_threshold  # Invalidate it from future selection
            node_colors[blue_node] = 2

    if can_finish:
        no_graph = Data(
            num_nodes=yes_graph.num_nodes,
            edge_index=torch.clone(yes_graph.edge_index),
            x=torch.from_numpy(node_colors[:, np.newaxis]).long(),
            y=torch.LongTensor([0]),
        )
        return no_graph
    else:
        return None


def modify_add_edge(yes_graph, hop_threshold, sp_table):
    """
    Adds a single edge, so that the new graph contains an additional RB pair within ``hop_threshold`` (or less).

    :param yes_graph: A graph that doesn't have a RB pair within a distance of ``hop_threshold``.
    :param hop_threshold: The classification threshold for the maximum distance.
    :param sp_table: A table with dimensions ``(N, N)`` that contains the shortest paths between each pair of nodes.

    :return: The desired "no" instance for the dataset.
    """
    closest_red = sp_table[
        :, yes_graph.x[:, 0] == 1
    ]  # This is of shape [num_nodes, num_red]
    closest_blue = sp_table[
        :, yes_graph.x[:, 0] == 2
    ]  # This is of shape [num_nodes, num_blue]

    num_red = closest_red.shape[1]
    num_blue = closest_blue.shape[1]
    num_nodes = closest_blue.shape[0]

    red_tiled = np.tile(
        closest_red, (num_nodes, num_blue)
    )  # [num_nodes^2, num_red * num_blue]
    # N1: R1 R2 R1 R2 R1 R2 ... N20: R1 R2 R1 R2, N1: R1 R2 R1 R2
    blue_repeated = np.repeat(closest_blue, num_nodes, axis=0)
    blue_repeated = np.repeat(
        blue_repeated, num_red, axis=1
    )  # [num_nodes^2, num_red * num_blue]
    # N1: B1 B1 B2 B2, N1: B1 B1 B2 B2, N2: B1 B1 B2 B2
    # Thus, by summing these we get the SP sum of the two nodes we are trying to connect. However,
    # this includes self-loops, which we must remove. We must also remove existing edges.
    sp_sums = red_tiled + blue_repeated
    self_loop_mask = [i * num_nodes + i for i in range(num_nodes)]
    existing_edge_mask = [
        yes_graph.edge_index[0, i].item() * num_nodes
        + yes_graph.edge_index[1, i].item()
        for i in range(yes_graph.edge_index.shape[1])
    ]
    overall_mask = self_loop_mask + existing_edge_mask
    overall_mask_as_bool = np.ones(num_nodes * num_nodes, dtype=bool)  # A 0/1 mask
    overall_mask_as_bool[
        overall_mask
    ] = False  # This mask will continue to be filtered as we go

    sp_sums_new_edges = sp_sums[overall_mask_as_bool, :]
    # We now have to check the possible edges against the clf_threshold
    min_sp_sums_new_edges = np.min(
        sp_sums_new_edges, axis=1
    )  # We now filter the edges where sum is >= thresh
    valid_sp_sums_mask = (
        min_sp_sums_new_edges < hop_threshold
    )  # i.e., thresh - 1 or less. Any would do.
    overall_mask_as_bool[
        overall_mask_as_bool
    ] = valid_sp_sums_mask  # Update the already true values
    # Now check if we have enough valid edges
    nb_valid_edges = np.sum(1 * overall_mask_as_bool)
    if nb_valid_edges >= 1:
        valid_idx = np.nonzero(1 * overall_mask_as_bool)[0]
        sample_edge = np.random.choice(valid_idx, 1, replace=False)[
            0
        ]  # That's the chosen edge
        edge_source = sample_edge // num_nodes
        edge_destination = (
            sample_edge % num_nodes
        )  # We now have the edge. Create the new graph.
        new_undir_edges = np.array(
            [[edge_source, edge_destination], [edge_destination, edge_source]]
        )
        # Create a copy of the yes graph here
        no_graph_x = np.array(yes_graph.x, copy=True)
        no_graph_edge_index = np.array(yes_graph.edge_index, copy=True)
        no_graph_edge_index = np.concatenate(
            (no_graph_edge_index, new_undir_edges), axis=1
        )
        # Note: There is no direct conversion from numpy
        #       to torch in torch_geometric.data.Data
        no_graph = Data(
            num_nodes=num_nodes,
            edge_index=torch.tensor(no_graph_edge_index),
            x=torch.tensor(no_graph_x),
            y=torch.LongTensor([0]),
        )
        return no_graph
    else:
        # No valid edges are possible. Cancel generation.
        return None
