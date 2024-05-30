import torch

from base import BaseDataLoader
from dataset.dataset import ECGDataset


class ECGDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size=8, shuffle=True, validation_split=0.2, num_workers=1, training=True):
        self.data_dir = data_dir
        dataset = ECGDataset(data_dir)
        print(len(dataset))
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)

