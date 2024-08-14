from data_provider.MyDataLoader import MyDataLoader
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter


def normalize_data(train_data, val_data, test_data):
    """标准化数据"""
    scaler = MinMaxScaler()
    scaler.fit(np.vstack(train_data))
    train_data_normalized = [scaler.transform(window.reshape(-1, window.shape[-1])).reshape(window.shape) for window
                             in
                             train_data]
    val_data_normalized = [scaler.transform(window.reshape(-1, window.shape[-1])).reshape(window.shape) for window
                           in
                           val_data]
    test_data_normalized = [scaler.transform(window.reshape(-1, window.shape[-1])).reshape(window.shape) for window
                            in
                            test_data]
    return train_data_normalized, val_data_normalized, test_data_normalized


def normalize_stats(train_stats, val_stats, test_stats):
    """标准化统计数据"""
    stats_scaler = MinMaxScaler()
    stats_scaler.fit(np.vstack(train_stats))
    train_stats_normalized = [stats_scaler.transform(stat.reshape(1, -1)) for stat in train_stats]
    val_stats_normalized = [stats_scaler.transform(stat.reshape(1, -1)) for stat in val_stats]
    test_stats_normalized = [stats_scaler.transform(stat.reshape(1, -1)) for stat in test_stats]
    return train_stats_normalized, val_stats_normalized, test_stats_normalized


def split_and_preprocess_data(windows, labels, windows_stats, test_size=0.2, val_size=0.1):
    # 标签编码
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    """划分并预处理数据集"""
    # 将数据划分为训练集、验证集和测试集
    train_indices, test_indices = train_test_split(np.arange(len(windows)), test_size=test_size, random_state=42)
    train_indices, val_indices = train_test_split(train_indices, test_size=val_size, random_state=42)

    train_windows = [windows[i] for i in train_indices]
    val_windows = [windows[i] for i in val_indices]
    test_windows = [windows[i] for i in test_indices]

    train_stats = [windows_stats[i] for i in train_indices]
    val_stats = [windows_stats[i] for i in val_indices]
    test_stats = [windows_stats[i] for i in test_indices]

    train_labels = [labels_encoded[i] for i in train_indices]
    val_labels = [labels_encoded[i] for i in val_indices]
    test_labels = [labels_encoded[i] for i in test_indices]

    # 标准化数据
    train_windows, val_windows, test_windows = normalize_data(train_windows, val_windows, test_windows)
    train_stats, val_stats, test_stats = normalize_stats(train_stats, val_stats, test_stats)

    return (train_windows, val_windows, test_windows,
            train_stats, val_stats, test_stats,
            train_labels, val_labels, test_labels)


def data_provider(args):
    batch_size = args.batch_size  # bsz for train and valid

    # 读数据（对此类进行改写）
    dataset = MyDataLoader(
        root_path=args.root_path,
        window_size=args.window_size,
        step_size=args.step_size
    )

    train_windows, val_windows, test_windows, train_stats, val_stats, test_stats, \
        train_labels, val_labels, test_labels = split_and_preprocess_data(dataset.windows, dataset.labels,
                                                                          dataset.windows_stats)

    train_dataset = windowsDataset(train_windows, train_stats, train_labels)
    valid_dataset = windowsDataset(val_windows, val_stats, val_labels)
    test_dataset = windowsDataset(test_windows, test_stats, test_labels)

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return dataset, train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader


class windowsDataset(Dataset):
    def __init__(self, windows, windows_stats, labels):
        self.windows = torch.tensor(windows, dtype=torch.float)
        windows_stats = torch.tensor(windows_stats, dtype=torch.float)
        self.windows_stats = windows_stats.view(-1, windows_stats.size(-1))
        self.labels = torch.tensor(labels, dtype=torch.long)

        self.label_counts = Counter(labels)
        self.total_samples = len(labels)
        self.label_to_index = self.build_label_to_index(labels)

    def build_label_to_index(self, labels):
        # 获取唯一标签并排序，然后分配索引
        unique_labels = sorted(set(labels))
        return {label: idx for idx, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.windows_stats[idx], self.labels[idx]
