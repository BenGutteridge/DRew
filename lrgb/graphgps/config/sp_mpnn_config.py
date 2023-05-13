from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


def set_cfg_sp_mpnn(cfg):
    r'''
    This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    '''

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    cfg.spn = CN()

    # then argument can be specified within the group
    cfg.spn.K = 0   # needs to be specified -- raises error if this comes back zero


register_config('sp_mpnn', set_cfg_sp_mpnn)
