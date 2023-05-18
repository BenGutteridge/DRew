
#!/bin/bash

### RingTransfer results

cd ../../..

cfg="configs/paper_configs/RINGTRANSFER/ring_transfer.yaml"

stage=stack
layer=gcnconv
nodes=(10 12 14 16 18 20 24 28 32 36 40 50 60 70)
for n in "${nodes[@]}"
do
  python main.py --cfg $cfg ring_dataset.num_nodes $n gnn.layers_mp $(($n/2)) device cuda gnn.stage_type $stage gnn.layer_type $layer
done

stage=drew_gnn
nu=1
layer=drew_gcnconv
nodes=(10 12 14 16 18 20 24 28 32 36 40 50 60 70)
for n in "${nodes[@]}"
do
  python main.py --cfg $cfg nu $nu ring_dataset.num_nodes $n gnn.layers_mp $(($n/2)) device cuda gnn.stage_type $stage gnn.layer_type $layer
done

stage=drew_gnn
nu=-1
layer=drew_gcnconv
nodes=(10 12 14 16 18 20 24 28 32 36 40 50 60 70)
for n in "${nodes[@]}"
do
  python main.py --cfg $cfg nu $nu ring_dataset.num_nodes $n gnn.layers_mp $(($n/2)) device cuda gnn.stage_type $stage gnn.layer_type $layer
done

stage=sp_gnn
layer=drew_gcnconv
nodes=(10 12 14 16 18 20 24 28 32 36 40 50 60 70)
for n in "${seeds[@]}"
do
  python main.py --cfg $cfg ring_dataset.num_nodes $n gnn.layers_mp $(($n/2)) device cuda gnn.stage_type $stage gnn.layer_type $layer
done