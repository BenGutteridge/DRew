from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


def set_cfg_ringtransfer(cfg):
    r'''
    Hyperparameters for ring transfer dataset (used when dataset.format is synthetic and dataset.name is RingTransfer)
    '''

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    cfg.ring_dataset = CN()
    cfg.ring_dataset.num_nodes = 10
    cfg.ring_dataset.num_graphs = 2000
    cfg.ring_dataset.num_classes = 5


register_config('ringtransfer', set_cfg_ringtransfer)
