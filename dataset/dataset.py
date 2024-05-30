import os
import csv
import matplotlib.pyplot as plt
import pywt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import torch
from torch.utils.data import Dataset, DataLoader


# Function to denoise data
def denoise(data):
    w = pywt.Wavelet('sym4')
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    threshold = 0.04  # Threshold for filtering

    coeffs = pywt.wavedec(data, 'sym4', level=maxlev)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))

    datarec = pywt.waverec(coeffs, 'sym4')

    return datarec


# PyTorch Dataset
class ECGDataset(Dataset):
    def __init__(self, data_dir, window_size=180, max_count=10000):
        self.data_dir = data_dir
        print(data_dir)
        self.window_size = window_size
        self.max_count = max_count
        self.classes = ['N', 'L', 'R', 'A', 'V']
        self.n_classes = len(self.classes)
        self.count_classes = [0] * self.n_classes

        self.X, self.y = self.load_data()
        self.X, self.y = self.balance_classes(self.X, self.y)

    def load_data(self):
        X = []
        y = []

        # Read files
        records, annotations = [], []
        for root, _, files in os.walk(self.data_dir):
            for f in files:
                if f.endswith('.csv'):
                    records.append(os.path.join(root, f))
                elif f.endswith('.txt'):
                    annotations.append(os.path.join(root, f))

        if not records or not annotations:
            raise FileNotFoundError("No CSV or TXT files found in the specified directory.")

        records.sort()
        annotations.sort()

        for r in range(len(records)):
            signals = []

            with open(records[r], 'rt') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                for row_index, row in enumerate(spamreader):
                    if row_index > 0:
                        signals.append(int(row[1]))

            signals = denoise(signals)
            signals = stats.zscore(signals)

            # Read annotations: R position and Arrhythmia class
            with open(annotations[r], 'r') as fileID:
                data = fileID.readlines()
                for d in range(1, len(data)):  # 0 index is Chart Head
                    splitted = list(filter(None, data[d].split(' ')))
                    pos = int(splitted[1])  # Sample ID
                    arrhythmia_type = splitted[2]  # Type
                    if arrhythmia_type in self.classes:
                        arrhythmia_index = self.classes.index(arrhythmia_type)
                        self.count_classes[arrhythmia_index] += 1
                        if self.window_size <= pos < (len(signals) - self.window_size):
                            beat = signals[pos - self.window_size:pos + self.window_size]
                            X.append(beat)
                            y.append(arrhythmia_index)

        return np.array(X), np.array(y)

    def balance_classes(self, X, y):
        df = pd.DataFrame(X)
        df['label'] = y

        dfs = [df[df['label'] == i] for i in range(self.n_classes)]
        balanced_dfs = [dfs[0].sample(n=5000, random_state=42)]
        for i in range(1, self.n_classes):
            balanced_dfs.append(resample(dfs[i], replace=True, n_samples=5000, random_state=42 + i))

        df_balanced = pd.concat(balanced_dfs)
        X_balanced = df_balanced.drop('label', axis=1).values
        y_balanced = df_balanced['label'].values

        return X_balanced, y_balanced

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(1), torch.tensor(self.y[idx], dtype=torch.long)
