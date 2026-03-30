import torch
import torch.nn as nn


class LSTM_MFCC(nn.Module):
    """LSTM для классификации MFCC-последовательностей"""
    
    def __init__(self, input_dim=13, hidden_dim=128, num_layers=2,
                 num_classes=2, dropout=0.3, bidirectional=True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        lstm_out, (hidden, _) = self.lstm(x)
        
        if self.bidirectional:
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            hidden_concat = torch.cat((hidden_forward, hidden_backward), dim=1)
            out = hidden_concat
        else:
            out = hidden[-1, :, :]
        
        return self.classifier(out)