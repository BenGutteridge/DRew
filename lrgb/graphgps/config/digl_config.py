from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


def set_cfg_digl(cfg):
    r'''
    Config for DIGL runs
    '''

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    # example argument group
    cfg.digl = CN()

    # then argument can be specified within the group
    cfg.digl.alpha = 0.15


register_config('digl', set_cfg_digl)
