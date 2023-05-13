import torch
from scipy.sparse.csgraph import floyd_warshall
from torch_geometric.utils import to_dense_adj, to_undirected
from collections import deque


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_shortest_paths(N, K, adj__N_N):
    # Computest the hops / jumps tensor for distance < K
    sp__N_N = floyd_warshall(adj__N_N)
    jumps__K_N_N = torch.zeros(size=(K, N, N), dtype=torch.float)
    for d in range(K):
        jumps__K_N_N[d] = torch.tensor(sp__N_N == d, dtype=torch.float)

    return jumps__K_N_N


def transform_graph_sp_small(graph, max_distance=None):
    if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
        original_edge_index, original_edge_attr = to_undirected(
            graph.edge_index, graph.edge_attr, reduce="max"
        )
    else:
        original_edge_index = to_undirected(graph.edge_index)

    adj_matrix = to_dense_adj(original_edge_index, max_num_nodes=graph.num_nodes)[0]
    shortest_paths = torch.tensor(floyd_warshall(adj_matrix), dtype=torch.long)
    edge_index = torch.zeros(
        size=(2, graph.num_nodes * graph.num_nodes), dtype=torch.long
    )
    edge_index[0, :] = torch.arange(graph.num_nodes).repeat(graph.num_nodes)
    edge_index[1, :] = torch.repeat_interleave(
        torch.arange(graph.num_nodes), graph.num_nodes
    )
    graph.edge_weights = shortest_paths.flatten()
    graph.edge_index = edge_index

    if max_distance:
        edge_mask_dist = graph.edge_weights <= max_distance
        graph.edge_weights = graph.edge_weights[edge_mask_dist]
        graph.edge_index = graph.edge_index[:, edge_mask_dist]

    if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
        mask_dist_1 = graph.edge_weights == 1
        graph.edge_index[:, mask_dist_1] = original_edge_index
        if len(graph.edge_attr.shape) == 1:
            graph.edge_attr = torch.zeros(
                size=(graph.edge_index.shape[1],), dtype=torch.long
            )
            graph.edge_attr[mask_dist_1] = original_edge_attr
        elif len(graph.edge_attr.shape) == 2:
            graph.edge_attr = torch.zeros(
                size=(graph.edge_index.shape[1], graph.edge_attr.shape[1]),
                dtype=original_edge_attr.dtype,
            )
            graph.edge_attr[mask_dist_1, :] = original_edge_attr

    return graph


def to_adj_list(N, edge_index, edge_attr=None):
    ret = [set() for _ in range(N)]
    for i in range(edge_index.shape[1]):
        u, v = edge_index[:, i]
        w = 0 if edge_attr is None else edge_attr[i]
        ret[u].add((v, w))
        ret[v].add((u, w))

    return ret


def transform_graph_sp(graph, max_distance=None):
    N = graph.num_nodes
    if max_distance is None:
        max_distance = N

    if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
        adj_list = to_adj_list(N, graph.edge_index, graph.edge_attr)
    else:
        adj_list = to_adj_list(N, graph.edge_index)

    edges = []
    weights = []
    edge_attr_new = []
    for src in range(N):
        # Create edges from src
        vis = set()
        vis.add(src)
        Q = deque([(src, 0)])
        while Q:
            u, d = Q.popleft()
            for v, w in adj_list[u]:
                if v not in vis:
                    vis.add(v)
                    edges.append([src, v])
                    weights.append(d + 1)
                    edge_attr_new.append(w)
                    if d + 1 < max_distance:
                        Q.append((v, d + 1))

    graph.edge_index = torch.tensor(edges, dtype=torch.long).T
    graph.edge_weights = torch.tensor(weights, dtype=torch.long)

    if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
        graph.edge_attr = torch.tensor(edge_attr_new, dtype=torch.long)

    return graph


class ShortestPathTransform:
    def __init__(self, max_distance=None, threshold=8196):
        self.max_distance = max_distance
        self.threshold = threshold

    def transform(self, graph):
        if graph.x is None:
            # For ogbg-ppa
            graph.x = torch.zeros(graph.num_nodes, dtype=torch.long)

        if graph.num_nodes > self.threshold or graph.num_edges == 0:
            return transform_graph_sp(graph, self.max_distance)
        else:
            return transform_graph_sp_small(graph, self.max_distance)
