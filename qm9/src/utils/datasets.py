import torch
import pickle
import os.path as osp
from torch_geometric.data import InMemoryDataset


class ProximityDataset(InMemoryDataset):
    def __init__(self, root, pre_transform=None, transform=None):
        super().__init__(root, pre_transform, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        raw_data = osp.join(self.root, "raw", "data_list.pickle")
        with open(raw_data, "rb") as f:
            data_list = pickle.load(f)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        torch.save(self.collate(data_list), self.processed_paths[0])
