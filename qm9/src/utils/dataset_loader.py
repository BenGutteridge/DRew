import json
import os
import os.path as osp
from utils.datasets import ProximityDataset
from torch_geometric.datasets import TUDataset
from utils.shortest_paths import ShortestPathTransform
import numpy as np
from torch_geometric.data import Data
from ogb.graphproppred import PygGraphPropPredDataset
import ogb
import torch
import gzip
import pickle


CHEMICAL = ["NCI1", "ENZYMES", "PROTEINS", "DD"]
OGB_GC = [
    "ogbg-molhiv",
    "ogbg-molpcba",
    "ogbg-ppa",
    "ogbg-code2",
    "ogbg-moltox21",
    "ogbg-moltoxcast",
    "ogbg-molbbbp",
    "ogbg-molbace",
    "ogbg-molmuv",
    "ogbg-molclintox",
    "ogbg-molsider",
]
ogb_metric = {
    "ogbg-molhiv": "rocauc",
    "ogbg-molpcba": "ap",
    "ogbg-ppa": "acc",
    "ogbg-code2": "F1",
    "ogbg-moltox21": "rocauc",
    "ogbg-moltoxcast": "rocauc",
    "ogbg-molbbbp": "rocauc",
    "ogbg-molbace": "rocauc",
    "ogbg-molmuv": "ap",
    "ogbg-molclintox": "rocauc",
    "ogbg-molsider": "rocauc",
}


def read_qm9(direc, file, transform_f, processed_root):
    path = osp.join(direc, file + ".jsonl.gz")  # Added pre-storing
    presaved_path = osp.join(processed_root, file + ".pre")
    if not osp.exists(presaved_path):  # The file doesn't exist
        with gzip.open(path, "r") as f:
            print('File not found. Reading from %s...' % path)
            data = f.read().decode("utf-8")
            graphs = [json.loads(jline) for jline in data.splitlines()]
            pyg_graphs = [
                transform_f(
                    map_qm9_to_pyg(graph, make_undirected=True, remove_dup=False)
                )
                for graph in graphs
            ]
            if not osp.exists(processed_root):
                os.mkdir(processed_root)
            with open(presaved_path, "wb") as g:  # Save for future reference
                print('Saving file to %s...' % presaved_path)
                pickle.dump(pyg_graphs, g)
                g.close()
            f.close()
            return pyg_graphs
    else:  # Load the pre-existing file
        with open(presaved_path, "rb") as g:
            print('Loading pre-saved file from %s...' % presaved_path)
            pyg_graphs = pickle.load(g)
            g.close()
        return pyg_graphs


def map_qm9_to_pyg(json_file, make_undirected=True, remove_dup=False):
    # We're making the graph undirected just like the original repo.
    # Note: make_undirected makes duplicate edges, so we need to preserve edge types.
    # Note: The original repo also add self-loops. We don't need that given how we see hops.
    edge_index = np.array([[g[0], g[2]] for g in json_file["graph"]]).T  # Edge Index
    edge_attributes = np.array(
        [g[1] - 1 for g in json_file["graph"]]
    )  # Edge type (-1 to put in [0, 3] range)
    if (
        make_undirected
    ):  # This will invariably cost us edge types because we reduce duplicates
        edge_index_reverse = edge_index[[1, 0], :]
        # Concat and remove duplicates
        if remove_dup:
            edge_index = torch.LongTensor(
                np.unique(
                    np.concatenate([edge_index, edge_index_reverse], axis=1), axis=1
                )
            )
        else:
            edge_index = torch.LongTensor(
                np.concatenate([edge_index, edge_index_reverse], axis=1)
            )
            edge_attributes = torch.LongTensor(
                np.concatenate([edge_attributes, np.copy(edge_attributes)], axis=0)
            )
    x = torch.FloatTensor(np.array(json_file["node_features"]))
    y = torch.FloatTensor(np.array(json_file["targets"]).T)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attributes, y=y)


def get_dataset(args, root_dir):
    dataset_path = osp.join(root_dir, "data", args.dataset)

    if args.mode == "gc" or args.mode == "gr":  # Graph Classification / Regression
        transform_class = ShortestPathTransform(max_distance=10)
    else:
        raise ValueError("Invalid mode.")

    # Load dataset
    if args.dataset in CHEMICAL:
        dataset = TUDataset(
            dataset_path,
            name=args.dataset,
            use_node_attr=True,
            pre_transform=transform_class.transform,
        )
        with open(
            osp.join(
                root_dir, "data_splits", "CHEMICAL", args.dataset + "_splits.json"
            ),
            "r",
        ) as f:
            splits = json.load(f)
        return dataset, splits, None
    elif args.dataset in OGB_GC:
        dataset = PygGraphPropPredDataset(
            name=args.dataset,
            root=dataset_path,
            pre_transform=transform_class.transform,
        )
        evaluator = ogb.graphproppred.Evaluator(args.dataset)
        metric = ogb_metric[args.dataset]
        return dataset, evaluator, metric
    elif args.dataset == "QM9":  # Graph Regression. This is re-computing it every time
        qm9_proc_root = osp.join(dataset_path, "QM9_proc")
        tr_graphs = read_qm9(
            dataset_path, "train", transform_class.transform, qm9_proc_root
        )
        val_graphs = read_qm9(
            dataset_path, "valid", transform_class.transform, qm9_proc_root
        )
        tst_graphs = read_qm9(
            dataset_path, "test", transform_class.transform, qm9_proc_root
        )
        num_feat = 15
        num_pred = 13  
        return tr_graphs, val_graphs, tst_graphs, num_feat, num_pred
    elif args.dataset.endswith("Prox"):
        dataset_path = osp.join(root_dir, "data", "Prox", args.dataset)
        dataset = ProximityDataset(
            root=dataset_path, pre_transform=transform_class.transform
        )
        with open(
            osp.join(root_dir, "data_splits", "Prox", args.dataset + "_splits.json"),
            "r",
        ) as f:
            splits = json.load(f)
        return dataset, splits, None
    else:
        raise ValueError("Invalid dataset.")
