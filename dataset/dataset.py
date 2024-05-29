import numpy as np
import wfdb
from torch.utils.data import Dataset


class OneHotEncoder:
    """188-bit one-hot encoder for ECG signal values."""
    def __init__(self, max_length=188):
        self.max_length = max_length

    def encode(self, signal):
        one_hot_encoded = np.zeros((len(signal), self.max_length))
        # Normalize and scale the signal indices into range [0, max_length-1]
        index = (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) * (self.max_length - 1)
        index = index.astype(int)
        one_hot_encoded[np.arange(len(signal)), index] = 1
        return one_hot_encoded


class ECGDataset(Dataset):
    def __init__(self, data_dir, record_numbers):
        self.signals = []
        self.labels = []
        self.encoder = OneHotEncoder()
        for record_number in record_numbers:
            record_path = f"{data_dir}/{record_number}"
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            encoded_signal = self.encoder.encode(record.p_signal[:, 0])
            self.signals.append(encoded_signal)
            self.labels.append(annotation.symbol)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]
