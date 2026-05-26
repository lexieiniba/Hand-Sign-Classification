import torch
import torch.nn as nn

class MarketLSTM(nn.Module):
    # 1. input_size is now 1 (because you are only passing Log Returns)
    def __init__(self, input_size=1, hidden_size=128, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        # 2. Simplified head for binary classification (Up vs Down)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # 3. Sigmoid forces the output between 0 and 1
        )

    def forward(self, x):
        # x shape: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # Take the last hidden state
        pred = self.head(last)
        return pred # Output shape: (batch, 1)