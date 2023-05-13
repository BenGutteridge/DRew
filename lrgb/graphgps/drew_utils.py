import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from tqdm import tqdm
from torch_geometric.graphgym.config import cfg
import os
from torch_geometric.data import Data

def get_task_id():
  if cfg.dataset.name.startswith('peptides'):
    return 'pept'
  elif cfg.dataset.format == 'PyG-VOCSuperpixels':
    return 'voc'
  elif cfg.dataset.format == 'PyG-COCOSuperpixels':
    return 'coco'
  elif cfg.dataset.name == 'PCQM4Mv2Contact-shuffle':
    return 'pcqm'
  else:
    raise NotImplementedError

default_heads = {
  'pept': 'graph',
  'voc': 'inductive_node',
}


def custom_set_out_dir(cfg, cfg_fname, name_tag, default=False):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = get_run_name(cfg_fname, default)
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def get_run_name(cfg_fname, default):
  dataset_name = ('-' + cfg.dataset.name) if cfg.dataset.name!='none' else ''
  if default:
    return os.path.splitext(os.path.basename(cfg_fname))[0]
  elif cfg.model.type == 'gnn':
    model = cfg.gnn.stage_type
  elif ('custom_gnn' in cfg.model.type) | (cfg.model.type=='drew_gated_gnn'):
    model = cfg.gnn.layer_type
  else:
    model = cfg.model.type
  if '+' in cfg.dataset.node_encoder_name: # note if PE used
    model += '_%s' % cfg.dataset.node_encoder_name.split('+')[-1]
  if cfg.model.type == 'mixhop_gcn':
    model += '_P=%02d' % cfg.mixhop_args.max_P
  if cfg.nu != 1:
    nu = '%02d' % cfg.nu if cfg.nu != -1 else 'inf'
    model += '_nu=%s' % nu
  if cfg.k_max < cfg.gnn.layers_mp:
    model += '_kmax=%02d' % cfg.k_max
  if cfg.agg_weights.use:
    model += '_weights'
  if cfg.agg_weights.convex_combo:
    model += '_CC'
  if cfg.dataset.transform != 'none':
    model += '_%s_alpha=p%02d' % (cfg.dataset.transform, cfg.digl.alpha*100)
  if cfg.spn.K != 0:
    model += '_K=%02d' % cfg.spn.K
  if cfg.gnn.batchnorm and cfg.model.type in ['drew_gin', 'mixhop_gcn']:
    model += '_bn'
  run_name = "%s%s_%s_bs=%04d_d=%03d_L=%02d" % (cfg.dataset.format, dataset_name, model, cfg.train.batch_size, cfg.gnn.dim_inner, cfg.gnn.layers_mp) # with BS
  # run_name = "%s%s_%s_d=%03d_L=%02d" % (cfg.dataset.format, dataset_name, model, cfg.gnn.dim_inner, cfg.gnn.layers_mp) # without BS
  cut = ['ides', 'ural', 'tional', 'PyG-', 'OGB-']
  for c in cut:
    run_name = run_name.replace(c, '')
  return run_name


def get_edge_labels(dataset):
  """takes in PyG dataset object and spits out some edge labels"""
  e = dataset.data.edge_attr
  if 'peptides' in dataset.root:
    print('Getting edge labels for peptides dataset...')
    # [bond_type, is_stereo, is_conj]
    edge_labels = e[:,0]*1 + e[:,2]*10 + e[:,1]*100  # columns
  elif 'QM9' in dataset.root:
    print('Getting edge labels for QM9 dataset')
    # one-hot vectors for bond type
    edge_labels = torch.argmax(e, dim=1)
  else:
    raise NotImplementedError("Dataset '%s' not supported" % dataset.folder)
  return edge_labels
