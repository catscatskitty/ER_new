import torch
import torch.nn as nn


class HybridSpectrogram(nn.Module):
    """Гибридная модель: CNN + LSTM на спектрограммах"""
    
    def __init__(self, n_mels=128, cnn_channels=64, lstm_hidden=128,
                 num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(32, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        
        self.lstm = nn.LSTM(
            input_size=cnn_channels * (n_mels // 4),  # Правильный расчёт размера
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        batch, channels, freq, time = x.size()
        
        # CNN
        x = self.cnn(x)  # (batch, cnn_channels, freq/4, time/4)
        
        # Изменяем форму для LSTM: (batch, time, features)
        # freq/4 - количество частотных бинов после пулинга
        # time/4 - количество временных шагов после пулинга
        batch, cnn_channels, freq_reduced, time_reduced = x.size()
        
        # Переставляем размерности: (batch, time, channels * freq)
        x = x.permute(0, 3, 1, 2)  # (batch, time, channels, freq)
        x = x.reshape(batch, time_reduced, cnn_channels * freq_reduced)
        
        # LSTM
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Берём последний выход LSTM
        out = lstm_out[:, -1, :]
        
        return self.classifier(out)