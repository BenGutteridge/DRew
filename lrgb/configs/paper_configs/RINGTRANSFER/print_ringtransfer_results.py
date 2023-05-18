import os
import os.path as osp
from os.path import join
import pandas as pd
from tabulate import tabulate


dir = input('Enter path to directory containing RingTransfer runs:\n')
table = []
for subdir in reversed(sorted(os.listdir(dir))):
    if 'RingTransfer' in subdir:
        try:
            df = pd.read_json(join(dir, subdir, 'agg', 'test', 'stats.json'), typ='series', lines=True)
        except:
            print('skipping %s' % subdir)
            break
        df = df[len(df)-1] # just final epoch
        final_acc, final_acc_std = df['accuracy'], df['accuracy_std']
        table.append([subdir, '%.2f ± %.2f' % (final_acc, final_acc_std)])
print(tabulate(table, headers=['Run', 'Accuracy'])) 