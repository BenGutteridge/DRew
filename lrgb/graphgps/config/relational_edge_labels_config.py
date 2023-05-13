from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


def set_cfg_edge_labels(cfg):
    r'''
    For utilising relational edge labels 
    '''

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    cfg.use_edge_labels = False
    cfg.edge_types = [] # fills in automatically
    cfg.edge_dim = 0 # fills in automatically


register_config('edge_labels', set_cfg_edge_labels)
