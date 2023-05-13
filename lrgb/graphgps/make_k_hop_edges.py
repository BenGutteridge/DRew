import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from tqdm import tqdm
from torch_geometric.graphgym.config import cfg
import os
from os.path import join, exists
from torch_geometric.data import Data

def make_k_hop_edges(dataset, K, format, name):
  print('Stage type %s, model type %s, using %d-hops' % (cfg.gnn.stage_type, cfg.model.type, K))
  # get k-hop edge amended dataset - either load or make it
  filedir = join(cfg.dataset.dir, 'k_hop_indices')    
  if not exists(filedir): os.makedirs(filedir) 

  # check if files exist already
  slic = '-slic=%02d' % cfg.dataset.slic_compactness if ((format == 'PyG-VOCSuperpixels') & (cfg.dataset.slic_compactness != 10)) else ''
  if cfg.dataset.transform != 'none':
    preproc = '-preproc=%s_alpha=p%02d' % (cfg.dataset.transform, int(100*cfg.digl.alpha))
  else: preproc = ''
  extra = ''.join([slic, preproc])
  file_exists = [exists(join(filedir, "%s-%s%s_k=%02d.pt" % (format, name, extra, k))) for k in range(1,K+1)] # list of K bools
  if not all(file_exists): # checks all files are there
    last_nonexistent_file = max(loc for loc, val in enumerate(file_exists) if val == False)+1
    print('Edge index file(s) not found for %s-%s%s_k=%02d (or lower); making file(s) now...' % (format, name, extra, last_nonexistent_file))
    compute_k_hop_edges(dataset, K, filedir, format, name, extra) # if they don't, make them

  # load files
  all_graphs = []
  print('Loading k-hop data files...')
  for k in tqdm(range(1,K+1)):
    filepath = join(filedir, "%s-%s%s_k=%02d.pt" % (format, name, extra, k))
    try:
      all_graphs.append(torch.load(filepath)) # [K,N,2,d]
    except: 
      files_to_remake = [k]
      for j in range(K, k, -1):
        try: 
          torch.load(join(filedir, "%s-%s_k=%02d.pt" % (format, name, k)))
        except:
          files_to_remake.append(j)
      print('Issue with following files, deleting and remaking them...\nk = ', files_to_remake)
      for j in files_to_remake:
        filepath = join(filedir, "%s-%s_k=%02d.pt" % (format, name, j))
        os.remove(filepath)
      compute_k_hop_edges(dataset, max(files_to_remake), filedir, format, name)
      filepath = join(filedir, "%s-%s_k=%02d.pt" % (format, name, k))
      all_graphs.append(torch.load(filepath)) # [K,N,2,d]

  all_hops = [list(n) for n in zip(*all_graphs)] # Transposing. n is graph; all_hops indexed by graph. [N,K,2,d]
  labels = []   # get k-hop labels
  for n in all_hops:
    for k, khop in enumerate(n,1):
      labels += [1*k]*khop.shape[-1]
  labels = torch.tensor(labels, dtype=torch.long)
  all_hops = [torch.cat(n, dim=1) for n in all_hops] # [K,2,d]
  count, ei_slices = 0 , [0]
  for d in all_hops:
    count += d.shape[-1]
    ei_slices.append(count)
  ei_slices = torch.tensor(ei_slices)
  all_hops = torch.cat(all_hops, dim=1) # [2,d]
  # set to dataset
  dataset.data.edge_index = all_hops
  dataset.data.edge_attr = labels
  dataset.slices['edge_index'] = dataset.slices['edge_attr'] = ei_slices

  print('Checking correct conversion...')
  count  = 0
  for i in tqdm(range(len(dataset))):
    if not torch.equal(dataset.get(i).edge_attr.float(), dataset.data.edge_attr[ei_slices[i]:ei_slices[i+1]].float()):
      # print('Graph %d not changed in dataset._data_list; setting manually' % i)
      count += 1
      dataset._data_list[i] = Data(x=dataset.get(i).x,
                                edge_index=dataset.data.edge_index[:, ei_slices[i]:ei_slices[i+1]],
                                edge_attr=dataset.data.edge_attr[ei_slices[i]:ei_slices[i+1]],
                                y=dataset.get(i).y)
    assert torch.equal(dataset.get(i).edge_attr, dataset.data.edge_attr[ei_slices[i]:ei_slices[i+1]]) # check that the conversion worked
  if count > 0: print('%d/%d graphs not changed in dataset._data_list; have been set manually' % (count, len(dataset))) # this is expected for VOC and COCO

  return dataset


def compute_k_hop_edges(dataset, K, filedir, format, name, extra):
  """take regular dataset, save k-hop edges"""
  # we're saving a list of k-hop edge indices
  edge_indices = dataset.data.edge_index
  slices = dataset.slices['edge_index']
  idxs = [[] for _ in range(K)]
  for i in tqdm(range(len(slices)-1)):
  # for edge_index in [edge_indices]:
    edge_index = edge_indices[:, slices[i]:slices[i+1]]
    idxs[0].append(edge_index) # 1-hop
    try:
      tmp = to_dense_adj(edge_index).float()
    except:
      print('Offending tensor:\nedge_index:\n', edge_index, '\nedge_index.shape:', edge_index.shape)
      adj = None # if it fails, set adj to None to force an errorx
    adj = tmp.to_sparse().float()
    matrices = [tmp]
    for k in range(2, K+1):
      tmp = torch.bmm(adj, tmp)
      for j in range(tmp.shape[-1]):
        tmp[0, j, j] = 0 # remove self-connections
      tmp = (tmp>0).float() # remove edge multiples
      for m in matrices:
        tmp -= m
      tmp = (tmp>0).float() # remove -ves, cancelled edges
      idx, _ = dense_to_sparse(tmp) # outputs int64, which we want
      matrices.append(tmp)
      idxs[k-1].append(idx)
  for k, ei_k in enumerate(idxs, 1):
    filepath = join(filedir, "%s-%s%s_k=%02d.pt" % (format, name, extra, k))
    if not exists(filepath):
      print('Saving edge indices for k=%d to %s...' % (k, filepath))
      torch.save(ei_k, filepath)