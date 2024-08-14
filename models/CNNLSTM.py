import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionLayer, self).__init__()
        self.feature_dim = feature_dim
        self.attention_weights = nn.Parameter(torch.randn(feature_dim, 1))

    def forward(self, x):
        # x shape: (batch_size, seq_length, feature_dim)
        scores = torch.matmul(x, self.attention_weights).squeeze(-1)  # (batch_size, seq_length)
        attn_weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (batch_size, seq_length, 1)
        weighted = torch.mul(x, attn_weights)  # Apply attention weights
        output = torch.sum(weighted, dim=1)  # Sum over the sequence
        return output


class HierarchicalAttentionNetwork(nn.Module):
    def __init__(self, attention_dim):
        super(HierarchicalAttentionNetwork, self).__init__()
        self.attention_dim = attention_dim
        self.W = nn.Parameter(torch.Tensor(attention_dim, attention_dim))
        self.b = nn.Parameter(torch.Tensor(attention_dim))
        self.u = nn.Parameter(torch.Tensor(attention_dim, 1))

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.W)
        nn.init.normal_(self.b)
        nn.init.normal_(self.u)

    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, attention_dim]
        uit = torch.tanh(torch.matmul(x, self.W) + self.b)
        ait = torch.matmul(uit, self.u).squeeze(-1)

        if mask is not None:
            ait = ait.masked_fill(mask == 0, -1e9)  # 将mask为0的位置设为很大的负数，避免这些位置的权重过大

        ait = F.softmax(ait, dim=1)
        weighted_input = x * ait.unsqueeze(-1)
        output = weighted_input.sum(dim=1)
        return output


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()

        self.conv1 = nn.Conv1d(configs.enc_in, 64, kernel_size=18, stride=1, padding=(10 - 1) // 2)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=(5 - 1) // 2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=(3 - 1) // 2)
        self.lstm1 = nn.LSTM(128, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, batch_first=True)
        self.attention = HierarchicalAttentionNetwork(64)  # 使用自定义注意力层
        self.output = nn.Linear(64, 2)  # 输出层

    def forward(self, x, win):
        # 调整x的形状为(batch_size, channels, time_steps)
        x = x.transpose(1, 2)  # 这行代码将维度从(batch, time_steps, channels)调整为(batch, channels, time_steps)
        x = F.elu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.dropout(x, 0.15)
        x = F.elu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.dropout(x, 0.15)
        x = F.elu(self.conv3(x))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.dropout(x, 0.15)
        x, (hn, cn) = self.lstm1(x.transpose(1, 2))
        x, (hn, cn) = self.lstm2(x)
        x = self.attention(x)
        x = self.output(x)
        return x

