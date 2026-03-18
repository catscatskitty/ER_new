import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    """
    CNN1D модель для классификации аудио
    """
    
    def __init__(self, input_dim=87, num_classes=2, dropout=0.5):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Первый сверточный блок
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        # Второй сверточный блок
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        # Третий сверточный блок
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        
        # После трех пулингов размер уменьшается в 8 раз
        # Нужно убедиться, что input_dim делится на 8
        reduced_dim = input_dim // 8
        conv_output_size = 256 * reduced_dim
        
        # Полносвязные слои
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Добавляем канальное измерение
        x = x.unsqueeze(1)  # [batch, 1, features]
        
        # Conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.dropout(x)
        
        # Conv2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.dropout(x)
        
        # Conv3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x