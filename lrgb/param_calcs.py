# parameter calculators

from ogb.utils.features import get_atom_feature_dims
from graphgps.encoder.voc_superpixels_encoder import VOC_node_input_dim
from torch_geometric.graphgym.config import cfg
from graphgps.drew_utils import get_task_id
sort_and_removes_dupes = lambda mylist : sorted(list(dict.fromkeys(mylist)))

def set_d_fixed_params(cfg):
  N = cfg.fixed_params.N
  if N > 0:
    cfg.gnn.dim_inner = return_hidden_dim(N)
    print('Hidden dim manually set to %d for fixed param count of %dk' % (cfg.gnn.dim_inner, int(N/1000)))
  elif cfg.dataset.name == 'RingTransfer':
    if cfg.gnn.stage_type == 'drew_gnn': # (L^2+L)/2 scaling for DRew
        n_layers, d = cfg.gnn.layers_mp, cfg.gnn.dim_inner
        cfg.gnn.dim_inner = round((2 * (d ** 2) / (n_layers+1)) ** 0.5) # sets param count to roughly the same for fixed L
    print('Using d = %d' % cfg.gnn.dim_inner)
  else:
    print('Using given hidden dim of %d' % cfg.gnn.dim_inner)

def get_k_neighbourhoods(t):
  sp_nbhs = list(range(1, min(t+1, cfg.k_max)+1))
  return sort_and_removes_dupes(sp_nbhs)

def get_num_fc_drew(L):
  """Base number of FC layers in DRew MP"""
  num_fc = 0
  assert cfg.k_max >= 0, 'Error: k_max < 0'
  for t in range(L):
    k_nbhs = get_k_neighbourhoods(t)
    toprint = ' '.join([str(i).ljust(2) if i in k_nbhs else 'X'.ljust(2) for i in range(1, k_nbhs[-1]+1)])
    print('\t%02d: %s' % (t, toprint))
    num_fc += len(k_nbhs)
  return num_fc

def return_hidden_dim(N):
  """Return hidden dimension for MPNN based on """
  # number of FC layers in message passing
  N *= 0.99 # a little spare
  L = cfg.gnn.layers_mp
  is_drew_gcn = any([
    (cfg.gnn.stage_type.startswith('drew_gnn') & (cfg.model.type == 'gnn')), 
  ])
  if is_drew_gcn:
    num_fc = get_num_fc_drew(L)
  elif cfg.model.type == 'mixhop_gcn':
    P = cfg.mixhop_args.max_P 
    num_fc = P + (L-1) * P**2
  elif cfg.model.type == 'drew_gin':
    num_fc = get_num_fc_drew(L) + L # MLP_s + MLP_k for {k}s
  elif cfg.gnn.stage_type == 'drew_share_gnn': # weight sharing - only one W mp per layer
    num_fc = L
  elif cfg.gnn.layer_type in ['share_drewgatedgcnconv', 'gatedgcnconv_noedge']:
    num_fc = 4*L # A,B,D,E (no C currently)
  elif cfg.gnn.layer_type == 'drewgatedgcnconv':
    num_fc = 2*L + get_num_fc_drew(L)*2 # A,D and B_{k},E_{k}
  elif cfg.gnn.layer_type == 'gatedgcnconv':
    num_fc = 5*L # A,B,C,D,E
  elif cfg.gnn.layer_type in 'gcnconv':
    num_fc = L
  elif cfg.gnn.stage_type == 'sp_gnn':
    num_fc = min(L, cfg.k_max) * L
  else:
    raise ValueError('Unknown stage/layer type combination; stage_type: {0}, layer_type: {1}'.format(cfg.gnn.stage_type, cfg.gnn.layer_type))


  # other params and summation
  post_mp = cfg.gnn.layers_post_mp - 1       # 2-layer MLP at end -- not counting final layer to num classes
  num_bn = cfg.gnn.batchnorm * num_fc        # batch norm layers
  if cfg.model.type == 'mixhop_gcn': num_bn = cfg.gnn.batchnorm * L * cfg.mixhop_args.max_P
  task = get_task_id()
  if task == 'pept':
    node_embed = sum(get_atom_feature_dims())
    head = 10 # number of classes -- 11 for struct, close enough
    d = solve_quadratic(num_fc+post_mp, num_bn+node_embed+head, -N)
  elif task == 'voc':
    node_embed = VOC_node_input_dim
    head = 21 # number of classes
    d = solve_quadratic(num_fc+post_mp, num_bn+node_embed+head+post_mp, -N)
  elif task == 'coco':
    node_embed = VOC_node_input_dim
    head = 81 # number of classes
    d = solve_quadratic(num_fc+post_mp, num_bn+node_embed+head+post_mp, -N)
  elif task == 'pcqm':
    node_embed = sum(get_atom_feature_dims())
    post_mp += 1 # head is a fc, post-mp layer
    d = solve_quadratic(num_fc+post_mp, num_bn+node_embed+post_mp, -N)
  else:
    raise ValueError('Unknown dataset format {0}'.format(cfg.dataset.format))
  
  return round(d)

def solve_quadratic(a,b,c):
  # Solve the quadratic equation ax**2 + bx + c = 0
  d = (b**2) - (4*a*c)
  # find two solutions
  sol1 = (-b-d**.5)/(2*a)
  sol2 = (-b+d**.5)/(2*a)
  if sol1 > 0 and sol2 < 0:
    return sol1
  else:
    return sol2

