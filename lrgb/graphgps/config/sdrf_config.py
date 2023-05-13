from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


def set_cfg_sdrf(cfg):
    r'''
    This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    '''

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    # example argument group
    cfg.sdrf = CN()

    # then argument can be specified within the group
    cfg.sdrf.use = False


register_config('sdrf', set_cfg_sdrf)
