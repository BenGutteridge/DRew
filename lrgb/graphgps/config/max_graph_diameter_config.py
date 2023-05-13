from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


def set_cfg_max_graph_diameter(cfg):

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    # max diameter, so we don't end up making weights that will never be used
    cfg.max_graph_diameter = 1000 # set in code to be the max graph diameter in the dataset


register_config('max_graph_diameter', set_cfg_max_graph_diameter)
