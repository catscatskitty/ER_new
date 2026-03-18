import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    """
    LSTM модель для классификации аудио
    """
    
    def __init__(self, input_size=87, hidden_size=128, 
                 num_layers=2, num_classes=2, 
                 dropout=0.3, bidirectional=True):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Проекция для увеличения размерности (опционально)
        self.projection = nn.Linear(input_size, hidden_size)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Direction multiplier
        self.direction_multiplier = 2 if bidirectional else 1
        
        # Полносвязные слои
        self.fc1 = nn.Linear(hidden_size * self.direction_multiplier, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: [batch, features]
        batch_size = x.size(0)
        
        # Проекция и добавление временного измерения
        x = self.projection(x)  # [batch, hidden_size]
        x = x.unsqueeze(1)      # [batch, 1, hidden_size]
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Используем последний скрытый слой
        if self.bidirectional:
            # Для bidirectional: [2, batch, hidden] -> [batch, 2*hidden]
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        # Полносвязные слои
        x = self.fc1(hidden)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x