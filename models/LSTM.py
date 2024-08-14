import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.n_hidden = 128
        self.n_inputs = 14
        self.n_classes = 2

        # Layers
        self.lstm = nn.LSTM(self.n_inputs, self.n_hidden, num_layers=2, batch_first=True)
        self.fc = nn.Linear(self.n_hidden, self.n_classes)

    def forward(self, feature_mat, window_features):
        # LSTM layer
        # feature_mat = feature_mat.view(-1, 64, self.n_inputs)
        lstm_out, (h_n, c_n) = self.lstm(feature_mat)
        lstm_last_output = lstm_out[:, -1, :]

        # Fully connected layer
        final_out = self.fc(lstm_last_output)
        return final_out