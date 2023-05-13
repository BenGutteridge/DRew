
#!/bin/bash

### SPN results

for i in {0..12}
do
  python main.py -d QM9 -m SP_RSUM_WEIGHT --max_distance 10 --num_layers 8 --specific_task $i --emb_dim 128 --nb_reruns 3
done


### DRew results

# experimental results in paper for {\nu=1}DRew obtained from different seeds and tasks running in parallel and aggregated manually 
seeds=(0 10 11)
for i in {0..12}
do
  for seed in "${seeds[@]}"
  do
    python main.py -d QM9 -m DRew_RSUM_WEIGHT --nu 1 --num_layers 8 --specific_task $i --emb_dim 95 --nb_reruns 1 --seed $seed
  done
done

# experimental results in paper for {\nu=\infty}DRew obtained from different seeds and tasks running in parallel and aggregated manually 
seeds=(0 1 2)
for i in {0..12}
do
  for seed in "${seeds[@]}"
  do
    python main.py -d QM9 -m DRew_RSUM_WEIGHT --nu "-1" --num_layers 8 --specific_task $i --emb_dim 95 --nb_reruns 1 --seed $seed
  done
done