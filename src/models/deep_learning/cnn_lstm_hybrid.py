import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridCNN_LSTM(nn.Module):
    """
    Гибридная CNN-LSTM модель
    """
    
    def __init__(self, input_dim=87, hidden_size=64, num_classes=2, dropout=0.3):
        super().__init__()
        
        # CNN часть - работает с 1D сигналом
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        # После двух пулингов размер уменьшается в 4 раза
        # input_dim=87 -> 87//4 = 21
        self.cnn_output_features = 64 * (input_dim // 4)  # 64 * 21 = 1344
        
        # Проекция для уменьшения размерности перед LSTM
        self.projection = nn.Linear(self.cnn_output_features, hidden_size)
        
        # LSTM часть
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Полносвязные слои
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: [batch, features] (87 признаков)
        batch_size = x.size(0)
        
        # Добавляем канальное измерение для CNN
        x = x.unsqueeze(1)  # [batch, 1, features]
        
        # CNN
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.dropout(x)
        
        # x shape после CNN: [batch, channels=64, features_reduced=21]
        
        # Изменяем форму: [batch, channels, features] -> [batch, features * channels]
        x = x.contiguous().view(batch_size, -1)  # [batch, 64*21 = 1344]
        
        # Применяем проекцию для уменьшения размерности
        x = self.projection(x)  # [batch, hidden_size=64]
        
        # Добавляем временное измерение для LSTM
        x = x.unsqueeze(1)  # [batch, 1, hidden_size]
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Используем последний скрытый слой (оба направления)
        # hidden shape: [2*num_layers, batch, hidden_size]
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)  # [batch, hidden_size*2]
        
        # Полносвязные слои
        x = self.fc1(hidden)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x