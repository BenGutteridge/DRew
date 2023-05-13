import torch
import numpy as np


def find_coloring(
    graph,
    sp_table,
    hop_threshold,
    num_red,
    num_tries=100,
    auxiliary_colors=8,
    count_threshold=3,
):
    """
    We use rejection sampling to find the coloring.

    :param graph: The graph in PyTorch Geometric format.
    :param sp_table: A table with dimensions ``(N, N)`` that contains the shortest paths between each pair of nodes.
    :param hop_threshold: The classification threshold for the maximum distance.
    :param count_threshold: The threshold number of blue nodes not supposed to appear within hop_thresh from red.
    :param num_red: Number of red labelled nodes.
    :param num_tries: Number of rejection sampling tries. If we couldn't generate a satisfying coloring,
                      we will return ``None``.
    :param auxiliary_colors: Number of auxiliary colors to add (this makes the task more challenging to learn).

    :return: A PyTorch Geometric graph with an additional attribute ``x`` of dimension ``(N, 1)`` where
             the values correspond to the colors:
                 0 - White
                 1 - Red
                 2 - Blue
                 3+ - Auxiliary Colors
    """
    assert num_red > 0
    assert num_red <= graph.num_nodes

    # We assume that we have fewer red vertices,
    # and if we don't, we will flip the colors in the
    # end of the procedure.
    for _ in range(num_tries):
        node_colors = np.zeros(graph.num_nodes)  # Initially assign white to all nodes.
        red_nodes = np.random.choice(graph.num_nodes, size=num_red, replace=False)
        node_colors[red_nodes] = 1
        # Venn-diagram like approach. Get within thresh of all red nodes
        within_thresh_mat = np.zeros((graph.num_nodes, num_red))  # [N, #Red]
        # Sequential generation
        for idx, red_node in enumerate(red_nodes):
            within_thresh_nodes = np.arange(graph.num_nodes)[
                sp_table[red_node, :] <= hop_threshold
            ]  # Get possible Bl
            within_thresh_mat[within_thresh_nodes, idx] = 1
        within_thresh_mat[red_nodes, :] = np.full(
            num_red, count_threshold
        )  # To avoid sampling a red node
        # (must be last)
        # We now have a matrix [N, #Red] saying which node is within hop_thresh of each red. We now sample from these
        running_thresh = np.zeros((1, num_red))
        target = np.full_like(
            running_thresh, count_threshold - 1
        )  # We want to have exactly count - 1 per red
        can_finish = True
        while np.max(target - running_thresh) > 0:  # While there is a node that works
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
                indices = np.arange(graph.num_nodes)[
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
        if not can_finish:
            continue  # Couldn't find the correct nodes

        # Now pick some random "outer nodes to make blue in the outer hop
        outer_nodes = np.sum(1 * within_thresh_mat, axis=1) == 0
        outer_idx = np.arange(graph.num_nodes)[outer_nodes]
        nb_outer_blue = np.random.randint(0, 4)  # 0, 1, 2, 3 possible
        if outer_idx.shape[0] >= nb_outer_blue:
            outer_blue_idx = np.random.choice(
                outer_idx, size=nb_outer_blue, replace=False
            )
            node_colors[outer_blue_idx] = 2

        # Additional step, use more colors
        non_colored_nodes = node_colors[node_colors == 0]
        if auxiliary_colors > 1:
            selection_range = [0] + [x for x in range(3, 2 + auxiliary_colors, 1)]
        else:
            selection_range = [0]
        other_colors = np.random.choice(
            selection_range, size=non_colored_nodes.shape[0], replace=True
        )
        node_colors[node_colors == 0] = other_colors
        graph.x = torch.from_numpy(node_colors[:, np.newaxis]).long()
        graph.y = torch.LongTensor([1])
        return graph

    return None
