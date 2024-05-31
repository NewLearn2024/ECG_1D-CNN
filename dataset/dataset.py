import os
import csv
import pywt
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.utils import resample
import torch
from torch.utils.data import Dataset


def denoise(data):
    w = pywt.Wavelet('sym4')
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    threshold = 0.04

    coeffs = pywt.wavedec(data, 'sym4', level=maxlev)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))

    datarec = pywt.waverec(coeffs, 'sym4')
    return datarec


class ECGDataset(Dataset):
    def __init__(self, data_dir, window_size=93, max_count=10000):
        self.data_dir = data_dir
        self.window_size = window_size
        self.max_count = max_count
        self.classes = ['N', 'L', 'R', 'A', 'V']
        self.n_classes = len(self.classes)
        self.count_classes = [0] * self.n_classes

        self.X, self.y = self.load_data()
        # self.plot_class_distribution(self.y, title="Initial Class Distribution")
        self.X, self.y = self.balance_classes(self.X, self.y)
        # self.plot_class_distribution(self.y, title="Balanced Class Distribution")
        print(self.X.shape, self.y.shape)

    def load_data(self):
        X = []
        y = []

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

            # # 시각화: 원본 신호
            # if r == 0:
            #     plt.figure(figsize=(10, 2))
            #     plt.plot(signals[:700], label='Original Signal')
            #     plt.legend()
            #     plt.show()

            # noise 제거
            signals = denoise(signals)

            # # 시각화: 잡음 제거 후 신호
            # if r == 0:
            #     plt.figure(figsize=(10, 2))
            #     plt.plot(signals[:700], label='Denoised Signal')
            #     plt.legend()
            #     plt.show()

            # 정규화
            signals = stats.zscore(signals)

            # # 시각화: 정규화 후 신호
            # if r == 0:
            #     plt.figure(figsize=(10, 2))
            #     plt.plot(signals[:700], label='Normalized Signal')
            #     plt.legend()
            #     plt.show()

            with open(annotations[r], 'r') as fileID:
                data = fileID.readlines()
                for d in range(1, len(data)):
                    splitted = list(filter(None, data[d].split(' ')))
                    pos = int(splitted[1])
                    arrhythmia_type = splitted[2]
                    if arrhythmia_type in self.classes:
                        arrhythmia_index = self.classes.index(arrhythmia_type)
                        self.count_classes[arrhythmia_index] += 1
                        if self.window_size <= pos < (len(signals) - self.window_size):
                            start_idx = pos - self.window_size
                            end_idx = pos + self.window_size + 1
                            X.append(signals[start_idx:end_idx])
                            y.append(arrhythmia_index)

                            # # 시각화: R-peak 중심의 비트
                            # if r == 0:
                            #     plt.figure(figsize=(10, 2))
                            #     plt.plot(beat, label='Beat Centered at R-peak')
                            #     plt.legend()
                            #     plt.show()

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

    def plot_class_distribution(self, y, title="Class Distribution"):
        plt.figure(figsize=(8, 6))
        values, counts = np.unique(y, return_counts=True)
        labels = [self.classes[v] for v in values]
        plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title(title)
        plt.axis('equal')
        plt.show()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

