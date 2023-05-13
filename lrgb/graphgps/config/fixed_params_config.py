from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


def set_cfg_fixed_params(cfg):
    r'''
    Sets the fixed parameter count that models must adhere to. Hidden dimension calculated automatically from this.
    If N = 0, the specified hidden dimension is used
    '''
    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    cfg.fixed_params = CN()
    cfg.fixed_params.N = 0 # a default, ignored if not >0

register_config('fixed_params', set_cfg_fixed_params)