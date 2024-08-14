import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.n_hidden = 128
        self.n_inputs = 14
        self.n_classes = 2

        # 定义GRU层
        self.gru = nn.GRU(self.n_inputs, self.n_hidden, batch_first=True)

        self.fc = nn.Linear(self.n_hidden, self.n_classes)

    def forward(self, feature_mat, window_features):
        # 初始化隐藏状态
        h0 = torch.zeros(1, feature_mat.size(0), self.n_hidden).to(feature_mat.device)

        # GRU 前向传播
        out, _ = self.gru(feature_mat, h0)

        # 取最后一个时间步的输出
        out = out[:, -1, :]

        # 全连接层
        out = self.fc(out)

        return out