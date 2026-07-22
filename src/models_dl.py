import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1DModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN1DModel, self).__init__()
        # Input: (Batch, 1, 82)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.4) 
        
        # 82 -> 41 after pool
        self.fc1 = nn.Linear(128 * 41, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class LSTMModel(nn.Module):
    def __init__(self, num_classes=2):
        super(LSTMModel, self).__init__()
        # Input: (Batch, 1, 82)
        self.lstm = nn.LSTM(input_size=82, hidden_size=64, num_layers=2, 
                            batch_first=True, bidirectional=True, dropout=0.2)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)

class HybridModel(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, 
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.transpose(1, 2) 
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)
