"""
Гибридная CNN-LSTM модель для классификации аудио (1D признаки)
Работает с акустическими (38), фонетическими (80) и комбинированными (118) признаками
"""

import torch
import torch.nn as nn


class CNNLSTMHybrid(nn.Module):
    """Гибридная модель: CNN + LSTM"""
    
    def __init__(self, input_dim, hidden_size=64, num_classes=2, dropout=0.3):
        """
        Args:
            input_dim: размерность входных признаков (38/80/118)
            hidden_size: размер скрытого состояния LSTM
            num_classes: количество классов
            dropout: вероятность dropout
        """
        super().__init__()
        
        # CNN часть для извлечения локальных паттернов
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # LSTM часть для временных зависимостей
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.cnn.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
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
        # x shape: (batch, input_dim)
        x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        # CNN
        x = self.cnn(x)  # (batch, 64, 1)
        x = x.squeeze(-1)  # (batch, 64)
        x = x.unsqueeze(1)  # (batch, 1, 64) для LSTM
        
        # LSTM
        lstm_out, (hidden, _) = self.lstm(x)
        hidden_concat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        
        # Классификация
        return self.classifier(hidden_concat)