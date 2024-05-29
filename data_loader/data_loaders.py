import wfdb

from base import BaseDataLoader
from dataset.dataset import ECGDataset


record_numbers = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]


class ECGDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size=8, shuffle=True, validation_split=0.04, num_workers=1):
        self.data_dir = data_dir
        dataset = ECGDataset(data_dir, record_numbers)
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)
