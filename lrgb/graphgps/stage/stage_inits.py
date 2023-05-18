import torch.nn as nn
# import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
# from torch_geometric.graphgym.register import register_stage
import torch
from .example import GNNLayer
from param_calcs import get_k_neighbourhoods

sort_and_removes_dupes = lambda mylist : sorted(list(dict.fromkeys(mylist)))

def init_DRewGCN(model, dim_in, dim_out, num_layers, skip_first_hop=False):
  """The (nu)DRew-GCN param initialiser, used for drew_gnn"""
  model.num_layers, use_weights = num_layers, cfg.agg_weights.use
  model.nu = cfg.nu if cfg.nu != -1 else float('inf')
  W_kt = {}
  if use_weights: alpha_t = []
  t0 = 1 if skip_first_hop else 0
  for t in range(t0, num_layers):
    d_in = dim_in if t == 0 else dim_out
    k_neighbourhoods = get_k_neighbourhoods(t)
    for k in k_neighbourhoods:
      W_kt["k=%d, t=%d" % (k,t)] = GNNLayer(d_in, dim_out) # regular GCN layers
    # if use_weights: alpha_t.append(torch.nn.Parameter(torch.randn(len(k_neighbourhoods)), requires_grad=True)) # random init from normal dist
    if use_weights: alpha_t.append(torch.nn.Parameter(torch.ones(len(k_neighbourhoods)), requires_grad=True)) # unity init
  model.W_kt = nn.ModuleDict(W_kt)
  if use_weights: model.alpha_t = nn.ParameterList(alpha_t)
  return model


def init_shareDRewGCN(model, dim_in, dim_out, num_layers, skip_first_hop=False):
  """The (nu)DRew-GCN param initialiser, but with weight sharing; used for drew_share_gnn"""
  model.num_layers, use_weights = num_layers, cfg.agg_weights.use
  model.nu = cfg.nu if cfg.nu != -1 else float('inf')
  W_t = {}
  if use_weights: alpha_t = []
  t0 = 1 if skip_first_hop else 0
  for t in range(t0, num_layers):
    d_in = dim_in if t == 0 else dim_out
    k_neighbourhoods = get_k_neighbourhoods(t)
    W_t["t=%d" % (t)] = GNNLayer(d_in, dim_out) # regular GCN layers
    # if use_weights: alpha_t.append(torch.nn.Parameter(torch.randn(len(k_neighbourhoods)), requires_grad=True)) # random init from normal dist
    if use_weights: alpha_t.append(torch.nn.Parameter(torch.ones(len(k_neighbourhoods)), requires_grad=True)) # unity init
  model.W_t = nn.ModuleDict(W_t)
  if use_weights: model.alpha_t = nn.ParameterList(alpha_t)
  return model


def init_SP_GCN(model, dim_in, dim_out, num_layers, max_k=None):
  """SP-GCN (with weight sharing) """
  assert num_layers == cfg.gnn.layers_mp
  model.num_layers = num_layers
  model.max_k = cfg.gnn.layers_mp if max_k is None else max_k
  if cfg.max_graph_diameter <= model.max_k:
    print("Warning: max_graph_diameter = %d; <= max_k, so setting max_k to max_graph_diameter" % cfg.max_graph_diameter)
    model.max_k = cfg.max_graph_diameter
  W, alpha_t = [], []
  for t in range(num_layers):
    W.append(GNNLayer(dim_in, dim_out)) # W_{k,t}
    alpha_t.append(torch.nn.Parameter(torch.randn(model.max_k), 
                                        requires_grad=True)) # random init from normal dist
  model.W = nn.ModuleList(W)
  model.alpha = nn.ParameterList(alpha_t)
  return model