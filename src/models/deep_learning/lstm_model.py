"""
LSTM модель для классификации аудио (1D признаки)
Работает с акустическими (38), фонетическими (80) и комбинированными (118) признаками
"""

import torch
import torch.nn as nn


class LSTMAudioClassifier(nn.Module):
    """LSTM для классификации на основе извлеченных признаков"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, 
                 num_classes=2, dropout=0.3, bidirectional=True):
        """
        Args:
            input_size: размерность входных признаков (38/80/118)
            hidden_size: размер скрытого состояния LSTM
            num_layers: количество слоёв LSTM
            num_classes: количество классов
            dropout: вероятность dropout
            bidirectional: использовать двунаправленную LSTM
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Проекция входных признаков в скрытое пространство
        self.projection = nn.Linear(input_size, hidden_size)
        
        # LSTM слои
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: (batch, input_size)
        x = x.unsqueeze(1)  # (batch, 1, input_size)
        x = self.projection(x)  # (batch, 1, hidden_size)
        
        # LSTM
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Используем последние скрытые состояния
        if self.bidirectional:
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            hidden_concat = torch.cat((hidden_forward, hidden_backward), dim=1)
            out = hidden_concat
        else:
            out = hidden[-1, :, :]
        
        return self.classifier(out)