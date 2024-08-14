import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.window_size
        self.label_len = configs.label_len
        self.pred_len = 0


        # FCN 部分
        self.conv1 = nn.Conv1d(in_channels=configs.enc_in, out_channels=128, kernel_size=8, padding='same')
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5,
                               padding='same')
        self.bn2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3,
                               padding='same')
        self.bn3 = nn.BatchNorm1d(128)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.projection = nn.Linear(128, 2)

    def classification(self, x_enc, window_features):

        # FCN 部分
        x = x_enc.permute(0, 2, 1)  # 将输入调整为 [batch_size, input_dim, seq_len]
        conv_out = F.relu(self.bn1(self.conv1(x)))
        conv_out = F.relu(self.bn2(self.conv2(conv_out)))
        conv_out = F.relu(self.bn3(self.conv3(conv_out)))
        conv_out = self.global_avg_pool(conv_out).squeeze(-1)  # 全局平均池化

        output = self.projection(conv_out)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, window_features):
        dec_out = self.classification(x_enc, window_features)
        return dec_out  # [B, N]
