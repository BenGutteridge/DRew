import logging
import os
import os.path as osp
import time
from functools import partial

import numpy as np
import torch
import torch_geometric.transforms as T
from numpy.random import default_rng
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import Data
from torch_geometric.datasets import (GNNBenchmarkDataset, Planetoid, TUDataset,
                                      WikipediaNetwork, ZINC, QM9)
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loader import load_pyg, load_ogb, set_dataset_attr
from torch_geometric.graphgym.register import register_loader

from graphgps.loader.dataset.coco_superpixels import COCOSuperpixels
from graphgps.loader.dataset.malnet_tiny import MalNetTiny
from graphgps.loader.dataset.voc_superpixels import VOCSuperpixels
from graphgps.loader.dataset.ring_transfer import RingTransferDataset
from graphgps.loader.split_generator import (prepare_splits,
                                             set_dataset_splits)
from graphgps.transform.posenc_stats import compute_posenc_stats
from graphgps.transform.transforms import (pre_transform_in_memory,
                                           typecast_x, concat_x_and_pos,
                                           clip_graphs_to_size)
from graphgps.drew_utils import get_edge_labels
from graphgps.make_k_hop_edges import make_k_hop_edges
from tqdm import tqdm


def log_loaded_dataset(dataset, format, name):
    logging.info(f"[*] Loaded dataset '{name}' from '{format}':")
    logging.info(f"  {dataset.data}")
    logging.info(f"  undirected: {dataset[0].is_undirected()}")
    logging.info(f"  num graphs: {len(dataset)}")

    total_num_nodes = 0
    if hasattr(dataset.data, 'num_nodes'):
        total_num_nodes = dataset.data.num_nodes
    elif hasattr(dataset.data, 'x'):
        total_num_nodes = dataset.data.x.size(0)
    logging.info(f"  avg num_nodes/graph: "
                 f"{total_num_nodes // len(dataset)}")
    logging.info(f"  num node features: {dataset.num_node_features}")
    logging.info(f"  num edge features: {dataset.num_edge_features}")
    if hasattr(dataset, 'num_tasks'):
        logging.info(f"  num tasks: {dataset.num_tasks}")

    if hasattr(dataset.data, 'y') and dataset.data.y is not None:
        if isinstance(dataset.data.y, list):
            # A special case for ogbg-code2 dataset.
            logging.info(f"  num classes: n/a")
        elif dataset.data.y.numel() == dataset.data.y.size(0) and \
                torch.is_floating_point(dataset.data.y):
            logging.info(f"  num classes: (appears to be a regression task)")
        else:
            logging.info(f"  num classes: {dataset.num_classes}")
    elif hasattr(dataset.data, 'train_edge_label') or hasattr(dataset.data, 'edge_label'):
        # Edge/link prediction task.
        if hasattr(dataset.data, 'train_edge_label'):
            labels = dataset.data.train_edge_label  # Transductive link task
        else:
            labels = dataset.data.edge_label  # Inductive link task
        if labels.numel() == labels.size(0) and \
                torch.is_floating_point(labels):
            logging.info(f"  num edge classes: (probably a regression task)")
        else:
            logging.info(f"  num edge classes: {len(torch.unique(labels))}")

    ## Show distribution of graph sizes.
    # graph_sizes = [d.num_nodes if hasattr(d, 'num_nodes') else d.x.shape[0]
    #                for d in dataset]
    # hist, bin_edges = np.histogram(np.array(graph_sizes), bins=10)
    # logging.info(f'   Graph size distribution:')
    # logging.info(f'     mean: {np.mean(graph_sizes)}')
    # for i, (start, end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
    #     logging.info(
    #         f'     bin {i}: [{start:.2f}, {end:.2f}]: '
    #         f'{hist[i]} ({hist[i] / hist.sum() * 100:.2f}%)'
    #     )


def load_dataset_master(format, name, dataset_dir):
    """
    Master loader that controls loading of all datasets, overshadowing execution
    of any default GraphGym dataset loader. Default GraphGym dataset loader are
    instead called from this function, the format keywords `PyG` and `OGB` are
    reserved for these default GraphGym loaders.

    Custom transforms and dataset splitting is applied to each loaded dataset.

    Args:
        format: dataset format name that identifies Dataset class
        name: dataset name to select from the class identified by `format`
        dataset_dir: path where to store the processed dataset

    Returns:
        PyG dataset object with applied perturbation transforms and data splits
    """
    if format.startswith('PyG-'):
        pyg_dataset_id = format.split('-', 1)[1]
        dataset_dir = osp.join(dataset_dir, pyg_dataset_id)

        if pyg_dataset_id == 'GNNBenchmarkDataset':
            dataset = preformat_GNNBenchmarkDataset(dataset_dir, name)

        elif pyg_dataset_id == 'MalNetTiny':
            dataset = preformat_MalNetTiny(dataset_dir, feature_set=name)

        elif pyg_dataset_id == 'Planetoid':
            dataset = Planetoid(dataset_dir, name)

        elif pyg_dataset_id == 'TUDataset':
            dataset = preformat_TUDataset(dataset_dir, name)

        elif pyg_dataset_id == 'VOCSuperpixels':
            dataset = preformat_VOCSuperpixels(dataset_dir, name,
                                               cfg.dataset.slic_compactness)

        elif pyg_dataset_id == 'COCOSuperpixels':
            dataset = preformat_COCOSuperpixels(dataset_dir, name,
                                                cfg.dataset.slic_compactness)


        elif pyg_dataset_id == 'WikipediaNetwork':
            if name == 'crocodile':
                raise NotImplementedError(f"crocodile not implemented yet")
            dataset = WikipediaNetwork(dataset_dir, name)

        elif pyg_dataset_id == 'ZINC':
            dataset = preformat_ZINC(dataset_dir, name)

        elif pyg_dataset_id == 'QM9':
            cfg.max_graph_diameter = 10
            dataset = preformat_QM9(dataset_dir)


        else:
            raise ValueError(f"Unexpected PyG Dataset identifier: {format}")

    # GraphGym default loader for Pytorch Geometric datasets
    elif format == 'PyG':
        dataset = load_pyg(name, dataset_dir)

    elif format == 'OGB':
        if name.startswith('ogbg'):
            dataset = preformat_OGB_Graph(dataset_dir, name.replace('_', '-'))

        elif name.startswith('PCQM4Mv2-'):
            subset = name.split('-', 1)[1]
            dataset = preformat_OGB_PCQM4Mv2(dataset_dir, subset)

        elif name.startswith('peptides-'):
            dataset = preformat_Peptides(dataset_dir, name)

        ### Link prediction datasets.
        elif name.startswith('ogbl-'):
            # GraphGym default loader.
            dataset = load_ogb(name, dataset_dir)
            # OGB link prediction datasets are binary classification tasks,
            # however the default loader creates float labels => convert to int.
            def convert_to_int(ds, prop):
                tmp = getattr(ds.data, prop).int()
                set_dataset_attr(ds, prop, tmp, len(tmp))
            convert_to_int(dataset, 'train_edge_label')
            convert_to_int(dataset, 'val_edge_label')
            convert_to_int(dataset, 'test_edge_label')

        elif name.startswith('PCQM4Mv2Contact-'):
            dataset = preformat_PCQM4Mv2Contact(dataset_dir, name)

        else:
            raise ValueError(f"Unsupported OGB(-derived) dataset: {name}")
    elif format == 'synthetic':
        if name=='RingTransfer':
            dataset = RingTransferDataset(num_graphs=cfg.ring_dataset.num_graphs,
                                            num_nodes=cfg.ring_dataset.num_nodes,
                                            num_classes=cfg.ring_dataset.num_classes)
        else: ValueError(f"Unknown data format/name: {format}, {name}")
    else:
        raise ValueError(f"Unknown data format: {format}")

    if cfg.sdrf.use:
        from gdl.data import SDRFDataset
        tmp = SDRFDataset(max_steps=5)
        dataset = tmp.process()

    multi_hop_stages = [
        'sp_gnn',
        'drew_gnn',
        'k_gnn',
    ]
    multi_hop_models = ['drew_gated_gnn', 'drew_gin']
    use_drew = any([
        (cfg.gnn.stage_type in multi_hop_stages),
        ('delay' in cfg.gnn.stage_type),
        (cfg.model.type in multi_hop_models),
    ])

    if cfg.dataset.transform.startswith('digl'):
        avg_degree = int(cfg.dataset.transform[cfg.dataset.transform.index('=')+1:])
        print('Using GDC transform, average degree %d' % avg_degree)
        alpha_str = '_alpha=p%02d'%int(cfg.digl.alpha*100) if cfg.digl.alpha != 0.15 else ''
        digl_filepath = osp.join(cfg.dataset.dir, 'k_hop_indices', '%s_%s_%s%s.pt' % (format, name, cfg.dataset.transform, alpha_str))
        if osp.exists(digl_filepath):
            print('Loading GDC transformed dataset from file %s' % digl_filepath)
            dataset = torch.load(digl_filepath)
        else:
            print('No file %s\nApplying transform...' % digl_filepath)
            tf = T.GDC(
                self_loop_weight=1.,
                normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=cfg.digl.alpha),
                sparsification_kwargs=dict(method='threshold', avg_degree=avg_degree),
                exact=True,
            ) # using default, except for avg degree
            tf = digl_tf_wrapper(tf)
            if 1 in dataset.data.edge_attr.shape:
                dataset = squeeze_edge_attrs(dataset)
            else:
                dataset = remove_edge_attrs(dataset)
            pre_transform_in_memory(dataset, tf, show_progress=True)
            try:
                print('Saving file to %s...' % digl_filepath)
                torch.save(dataset, digl_filepath)
            except: print('Failed to save.') # TODO: make this work
        if use_drew or ('noedge' in cfg.gnn.layer_type) or ('peptides' in name):
            dataset = remove_edge_attrs(dataset)
        else:
            dataset = unsqueeze_edge_attrs(dataset)

    if cfg.use_edge_labels:
        edge_labels = get_edge_labels(dataset)
        is_edge_str = '_edge_types'
        edge_types = [int(torch.unique(edge_labels)[i]) for i in range(len(torch.unique(edge_labels)))]
        n_digits = len(str(max(edge_types)))
        cfg.edge_types = [str(i).zfill(n_digits) for i in edge_types]
    else:
        edge_labels = None
        is_edge_str = ''

    if use_drew:
        k_max = min(cfg.gnn.layers_mp, cfg.k_max)
        dataset = make_k_hop_edges(dataset, k_max, format, name)
    log_loaded_dataset(dataset, format, name)

    # Precompute necessary statistics for positional encodings.
    pe_enabled_list = []
    for key, pecfg in cfg.items():
        if key.startswith('posenc_') and pecfg.enable:
            pe_name = key.split('_', 1)[1]
            pe_enabled_list.append(pe_name)
            if hasattr(pecfg, 'kernel'):
                # Generate kernel times if functional snippet is set.
                if pecfg.kernel.times_func:
                    pecfg.kernel.times = list(eval(pecfg.kernel.times_func))
                logging.info(f"Parsed {pe_name} PE kernel times / steps: "
                             f"{pecfg.kernel.times}")
    if pe_enabled_list:
        start = time.perf_counter()
        logging.info(f"Precomputing Positional Encoding statistics: "
                     f"{pe_enabled_list} for all graphs...")
        # Estimate directedness based on 10 graphs to save time.
        is_undirected = all(d.is_undirected() for d in dataset[:10])
        logging.info(f"  ...estimated to be undirected: {is_undirected}")
        pre_transform_in_memory(dataset,
                                partial(compute_posenc_stats,
                                        pe_types=pe_enabled_list,
                                        is_undirected=is_undirected,
                                        cfg=cfg),
                                show_progress=True
                                )
        elapsed = time.perf_counter() - start
        timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                  + f'{elapsed:.2f}'[-3:]
        logging.info(f"Done! Took {timestr}")

    # Set standard dataset train/val/test splits
    if hasattr(dataset, 'split_idxs'):
        set_dataset_splits(dataset, dataset.split_idxs)
        delattr(dataset, 'split_idxs')

    # Verify or generate dataset train/val/test splits
    prepare_splits(dataset)

    # Precompute in-degree histogram if needed for PNAConv.
    if cfg.gt.layer_type.startswith('PNAConv') and len(cfg.gt.pna_degrees) == 0:
        cfg.gt.pna_degrees = compute_indegree_histogram(
            dataset[dataset.data['train_graph_index']])

    return dataset


register_loader('custom_master_loader', load_dataset_master)


def compute_indegree_histogram(dataset):
    """Compute histogram of in-degree of nodes needed for PNAConv.

    Args:
        dataset: PyG Dataset object

    Returns:
        List where i-th value is the number of nodes with in-degree equal to `i`
    """
    from torch_geometric.utils import degree

    deg = torch.zeros(1000, dtype=torch.long)
    max_degree = 0
    for data in dataset:
        d = degree(data.edge_index[1],
                   num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, d.max().item())
        deg += torch.bincount(d, minlength=deg.numel())
    return deg.numpy().tolist()[:max_degree + 1]


def preformat_GNNBenchmarkDataset(dataset_dir, name):
    """Load and preformat datasets from PyG's GNNBenchmarkDataset.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific dataset in the TUDataset class

    Returns:
        PyG dataset object
    """
    tf_list = []
    if name in ['MNIST', 'CIFAR10']:
        tf_list = [concat_x_and_pos]  # concat pixel value and pos. coordinate
        tf_list.append(partial(typecast_x, type_str='float'))
    else:
        ValueError(f"Loading dataset '{name}' from "
                   f"GNNBenchmarkDataset is not supported.")

    dataset = join_dataset_splits(
        [GNNBenchmarkDataset(root=dataset_dir, name=name, split=split)
         for split in ['train', 'val', 'test']]
    )
    pre_transform_in_memory(dataset, T.Compose(tf_list))

    return dataset


def preformat_MalNetTiny(dataset_dir, feature_set):
    """Load and preformat Tiny version (5k graphs) of MalNet

    Args:
        dataset_dir: path where to store the cached dataset
        feature_set: select what node features to precompute as MalNet
            originally doesn't have any node nor edge features

    Returns:
        PyG dataset object
    """
    if feature_set in ['none', 'Constant']:
        tf = T.Constant()
    elif feature_set == 'OneHotDegree':
        tf = T.OneHotDegree()
    elif feature_set == 'LocalDegreeProfile':
        tf = T.LocalDegreeProfile()
    else:
        raise ValueError(f"Unexpected transform function: {feature_set}")

    dataset = MalNetTiny(dataset_dir)
    dataset.name = 'MalNetTiny'
    logging.info(f'Computing "{feature_set}" node features for MalNetTiny.')
    pre_transform_in_memory(dataset, tf)

    split_dict = dataset.get_idx_split()
    dataset.split_idxs = [split_dict['train'],
                          split_dict['valid'],
                          split_dict['test']]

    return dataset


def preformat_OGB_Graph(dataset_dir, name):
    """Load and preformat OGB Graph Property Prediction datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific OGB Graph dataset

    Returns:
        PyG dataset object
    """
    dataset = PygGraphPropPredDataset(name=name, root=dataset_dir)
    s_dict = dataset.get_idx_split()
    dataset.split_idxs = [s_dict[s] for s in ['train', 'valid', 'test']]

    if name == 'ogbg-ppa':
        # ogbg-ppa doesn't have any node features, therefore add zeros but do
        # so dynamically as a 'transform' and not as a cached 'pre-transform'
        # because the dataset is big (~38.5M nodes), already taking ~31GB space
        def add_zeros(data):
            data.x = torch.zeros(data.num_nodes, dtype=torch.long)
            return data
        dataset.transform = add_zeros
    elif name == 'ogbg-code2':
        from graphgps.loader.ogbg_code2_utils import idx2vocab, \
            get_vocab_mapping, augment_edge, encode_y_to_arr
        num_vocab = 5000  # The number of vocabulary used for sequence prediction
        max_seq_len = 5  # The maximum sequence length to predict

        seq_len_list = np.array([len(seq) for seq in dataset.data.y])
        logging.info(f"Target sequences less or equal to {max_seq_len} is "
            f"{np.sum(seq_len_list <= max_seq_len) / len(seq_len_list)}")

        # Building vocabulary for sequence prediction. Only use training data.
        vocab2idx, idx2vocab_local = get_vocab_mapping(
            [dataset.data.y[i] for i in s_dict['train']], num_vocab)
        logging.info(f"Final size of vocabulary is {len(vocab2idx)}")
        idx2vocab.extend(idx2vocab_local)  # Set to global variable to later access in CustomLogger

        # Set the transform function:
        # augment_edge: add next-token edge as well as inverse edges. add edge attributes.
        # encode_y_to_arr: add y_arr to PyG data object, indicating the array repres
        dataset.transform = T.Compose(
            [augment_edge,
             lambda data: encode_y_to_arr(data, vocab2idx, max_seq_len)])

        # Subset graphs to a maximum size (number of nodes) limit.
        pre_transform_in_memory(dataset, partial(clip_graphs_to_size,
                                                 size_limit=1000))

    return dataset


def preformat_OGB_PCQM4Mv2(dataset_dir, name):
    """Load and preformat PCQM4Mv2 from OGB LSC.

    OGB-LSC provides 4 data index splits:
    2 with labeled molecules: 'train', 'valid' meant for training and dev
    2 unlabeled: 'test-dev', 'test-challenge' for the LSC challenge submission

    We will take random 150k from 'train' and make it a validation set and
    use the original 'valid' as our testing set.

    Note: PygPCQM4Mv2Dataset requires rdkit

    Args:
        dataset_dir: path where to store the cached dataset
        name: select 'subset' or 'full' version of the training set

    Returns:
        PyG dataset object
    """
    try:
        # Load locally to avoid RDKit dependency until necessary.
        from ogb.lsc import PygPCQM4Mv2Dataset
    except Exception as e:
        logging.error('ERROR: Failed to import PygPCQM4Mv2Dataset, '
                      'make sure RDKit is installed.')
        raise e


    dataset = PygPCQM4Mv2Dataset(root=dataset_dir)
    split_idx = dataset.get_idx_split()

    rng = default_rng(seed=42)
    train_idx = rng.permutation(split_idx['train'].numpy())
    train_idx = torch.from_numpy(train_idx)

    # Leave out 150k graphs for a new validation set.
    valid_idx, train_idx = train_idx[:150000], train_idx[150000:]
    if name == 'full':
        split_idxs = [train_idx,  # Subset of original 'train'.
                      valid_idx,  # Subset of original 'train' as validation set.
                      split_idx['valid']  # The original 'valid' as testing set.
                      ]
    elif name == 'subset':
        # Further subset the training set for faster debugging.
        subset_ratio = 0.1
        subtrain_idx = train_idx[:int(subset_ratio * len(train_idx))]
        subvalid_idx = valid_idx[:50000]
        subtest_idx = split_idx['valid']  # The original 'valid' as testing set.
        dataset = dataset[torch.cat([subtrain_idx, subvalid_idx, subtest_idx])]
        n1, n2, n3 = len(subtrain_idx), len(subvalid_idx), len(subtest_idx)
        split_idxs = [list(range(n1)),
                      list(range(n1, n1 + n2)),
                      list(range(n1 + n2, n1 + n2 + n3))]
    else:
        raise ValueError(f'Unexpected OGB PCQM4Mv2 subset choice: {name}')
    dataset.split_idxs = split_idxs
    return dataset


def preformat_PCQM4Mv2Contact(dataset_dir, name):
    """Load PCQM4Mv2-derived molecular contact link prediction dataset.

    Note: This dataset requires RDKit dependency!

    Args:
       dataset_dir: path where to store the cached dataset
       name: the type of dataset split: 'shuffle', 'num-atoms'

    Returns:
       PyG dataset object
    """
    try:
        # Load locally to avoid RDKit dependency until necessary
        from graphgps.loader.dataset.pcqm4mv2_contact import \
            PygPCQM4Mv2ContactDataset, \
            structured_neg_sampling_transform
    except Exception as e:
        logging.error('ERROR: Failed to import PygPCQM4Mv2ContactDataset, '
                      'make sure RDKit is installed.')
        raise e

    split_name = name.split('-', 1)[1]
    dataset = PygPCQM4Mv2ContactDataset(dataset_dir)
    # Inductive graph-level split (there is no train/test edge split).
    s_dict = dataset.get_idx_split(split_name)
    dataset.split_idxs = [s_dict[s] for s in ['train', 'val', 'test']]
    if cfg.dataset.resample_negative:
        dataset.transform = structured_neg_sampling_transform
    return dataset


def preformat_Peptides(dataset_dir, name):
    """Load Peptides dataset, functional or structural.

    Note: This dataset requires RDKit dependency!

    Args:
        dataset_dir: path where to store the cached dataset
        name: the type of dataset split:
            - 'peptides-functional' (10-task classification)
            - 'peptides-structural' (11-task regression)

    Returns:
        PyG dataset object
    """
    try:
        # Load locally to avoid RDKit dependency until necessary.
        from graphgps.loader.dataset.peptides_functional import \
            PeptidesFunctionalDataset
        from graphgps.loader.dataset.peptides_structural import \
            PeptidesStructuralDataset
    except Exception as e:
        logging.error('ERROR: Failed to import Peptides dataset class, '
                      'make sure RDKit is installed.')
        raise e

    dataset_type = name.split('-', 1)[1]
    if dataset_type == 'functional':
        dataset = PeptidesFunctionalDataset(dataset_dir)
    elif dataset_type == 'structural':
        dataset = PeptidesStructuralDataset(dataset_dir)
    s_dict = dataset.get_idx_split()
    dataset.split_idxs = [s_dict[s] for s in ['train', 'val', 'test']]
    return dataset


def preformat_TUDataset(dataset_dir, name):
    """Load and preformat datasets from PyG's TUDataset.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific dataset in the TUDataset class

    Returns:
        PyG dataset object
    """
    if name in ['DD', 'NCI1', 'ENZYMES', 'PROTEINS']:
        func = None
    elif name.startswith('IMDB-') or name == "COLLAB":
        func = T.Constant()
    else:
        ValueError(f"Loading dataset '{name}' from TUDataset is not supported.")
    dataset = TUDataset(dataset_dir, name, pre_transform=func)
    return dataset


def preformat_ZINC(dataset_dir, name):
    """Load and preformat ZINC datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: select 'subset' or 'full' version of ZINC

    Returns:
        PyG dataset object
    """
    if name not in ['subset', 'full']:
        raise ValueError(f"Unexpected subset choice for ZINC dataset: {name}")
    dataset = join_dataset_splits(
        [ZINC(root=dataset_dir, subset=(name == 'subset'), split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def preformat_QM9(dataset_dir):
    """Load and preformat ZINC datasets.

    Args:
        dataset_dir: path where to store the cached dataset

    Returns:
        PyG dataset object
    """
    # if name not in ['subset', 'full']:
    #     raise ValueError(f"Unexpected subset choice for ZINC dataset: {name}")
    # dataset = join_dataset_splits(
    #     [ZINC(root=dataset_dir, subset=(name == 'subset'), split=split)
    #      for split in ['train', 'val', 'test']]
    # )

    # cfg.dataset.regression_targets = ['mu', 'alpha', 'HOMO', 'LUMO', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv', 'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'rot_A', 'rot_B', 'rot_C']
    cfg.edge_dim = 4
    return QM9(root=dataset_dir)


def preformat_VOCSuperpixels(dataset_dir, name, slic_compactness):
    """Load and preformat VOCSuperpixels dataset.

    Args:
        dataset_dir: path where to store the cached dataset
    Returns:
        PyG dataset object
    """
    dataset = join_dataset_splits(
        [VOCSuperpixels(root=dataset_dir, name=name,
                        slic_compactness=slic_compactness,
                        split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def preformat_COCOSuperpixels(dataset_dir, name, slic_compactness):
    """Load and preformat COCOSuperpixels dataset.

    Args:
        dataset_dir: path where to store the cached dataset
    Returns:
        PyG dataset object
    """
    dataset = join_dataset_splits(
        [COCOSuperpixels(root=dataset_dir, name=name,
                         slic_compactness=slic_compactness,
                         split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def join_dataset_splits(datasets):
    """Join train, val, test datasets into one dataset object.

    Args:
        datasets: list of 3 PyG datasets to merge

    Returns:
        joint dataset with `split_idxs` property storing the split indices
    """
    assert len(datasets) == 3, "Expecting train, val, test datasets"

    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])
    data_list = [datasets[0].get(i) for i in range(n1)] + \
                [datasets[1].get(i) for i in range(n2)] + \
                [datasets[2].get(i) for i in range(n3)]

    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    split_idxs = [list(range(n1)),
                  list(range(n1, n1 + n2)),
                  list(range(n1 + n2, n1 + n2 + n3))]
    datasets[0].split_idxs = split_idxs

    return datasets[0]

def remove_edge_attrs(dataset):
    """Removes edge attrs from dataset for experiments which don't use them"""
    dataset.data.edge_attr = None
    if any([dataset.get(i).edge_attr is not None for i in range(len(dataset))]):
        print('Removing edge attrs so GDC preprocessing can be performed...')
        count = 0
        for i in tqdm(range(len(dataset))): 
            if dataset.get(i).edge_attr is not None:
                count += 1
                dataset._data_list[i] = Data(x=dataset.get(i).x,
                                            edge_index=dataset.get(i).edge_index,
                                            edge_attr=None,
                                            y=dataset.get(i).y)
        assert not any([dataset.get(i).edge_attr is not None for i in range(len(dataset))])
    return dataset

def squeeze_edge_attrs(dataset):
    """Removes edge attrs from dataset for experiments which don't use them"""
    dataset.data.edge_attr = dataset.data.edge_attr.squeeze()
    if any([dataset.get(i).edge_attr.dim() > 1 for i in range(len(dataset))]):
        print('Squeezing edge attrs down to 1D...')
        count = 0
        for i in tqdm(range(len(dataset))): 
            if dataset.get(i).edge_attr.dim() > 1:
                count += 1
                dataset._data_list[i] = Data(x=dataset.get(i).x,
                                            edge_index=dataset.get(i).edge_index,
                                            edge_attr=dataset.get(i).edge_attr.squeeze(),
                                            y=dataset.get(i).y)
        assert not any([dataset.get(i).edge_attr.dim() > 1 for i in range(len(dataset))])
    return dataset

def unsqueeze_edge_attrs(dataset):
    """Turns back into nx1 2d tensor (ie col vector)"""
    dataset.data.edge_attr = dataset.data.edge_attr.reshape(-1,1)
    if any([dataset.get(i).edge_attr.dim() == 1 for i in range(len(dataset))]):
        print('Reshaping edge attrs...')
        for i in tqdm(range(len(dataset))): 
            if dataset.get(i).edge_attr.dim() == 1:
                d_i = dataset.get(i)
                dataset._data_list[i] = Data(x=d_i.x,
                                            edge_index=d_i.edge_index,
                                            edge_attr=d_i.edge_attr.reshape(-1,1),
                                            y=d_i.y,
                                            edge_index_labeled=getattr(d_i, 'edge_index_labeled', None),  # for PCQM
                                            edge_label=getattr(d_i, 'edge_label', None)                 # for PCQM
                                            )
        assert all([dataset.get(i).edge_attr.dim() == 2 for i in range(len(dataset))])
    return dataset

def digl_tf_wrapper(f):
    """For DIGL transformation; skipping over graphs with no edges"""
    def wrapper(x):
        if x.edge_index.shape[-1] == 0:
            print('Edgeless graph skipped.')
            if x.edge_attr is None:
                x = Data(edge_index=x.edge_index,
                        x=x.x,
                        edge_index_labeled=x.edge_index_labeled,
                        edge_label=x.edge_label,
                        edge_attr=torch.tensor([])
                        )
            return x
        else:
            return f(x)
    return wrapper