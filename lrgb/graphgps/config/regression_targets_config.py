from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


def set_cfg_regression_targets(cfg):
    r'''
    This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    '''

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    # names of individual regression targets
    cfg.dataset.regression_targets = [] # set in code or defaults to numbered list

register_config('regression_targets', set_cfg_regression_targets)
