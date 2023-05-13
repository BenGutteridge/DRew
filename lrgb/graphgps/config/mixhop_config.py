from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


def set_cfg_mixhop(cfg):
    r'''
    Config for MixHop-GCN experiments
    '''

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    # example argument group
    cfg.mixhop_args = CN()

    cfg.mixhop_args.layers = [200,200,200]
    cfg.mixhop_args.max_P = 3


register_config('mixhop_args', set_cfg_mixhop)
