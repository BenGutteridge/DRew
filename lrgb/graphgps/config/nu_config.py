from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


def set_cfg_nu(cfg):
    r'''
    Config for the DRew rate parameter \nu and a few other params
    '''

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    # example argument
    cfg.nu = 1
    
    cfg.k_max = int(1e6)    # Sets the max value of k for which k-neighbourhood aggregation is performed in DRew and SP-GCN. 1e6 used as default stand-in for infinity, i.e. no outer limit

    cfg.agg_weights = CN()
    cfg.agg_weights.use = False # determines whether to use learned weights for k-hop aggregations
    cfg.agg_weights.convex_combo = False # determines whether to use convex combination of weights -- weights are equal if '.use'=False


register_config('nu', set_cfg_nu)
