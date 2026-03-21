"""
CNN модель для классификации аудио (1D признаки)
Работает с акустическими (38), фонетическими (80) и комбинированными (118) признаками
"""

import torch
import torch.nn as nn


class CNNAudioClassifier(nn.Module):
    """1D CNN для классификации на основе извлеченных признаков"""
    
    def __init__(self, input_dim, num_classes=2, dropout=0.5):
        """
        Args:
            input_dim: размерность входных признаков (38/80/118)
            num_classes: количество классов
            dropout: вероятность dropout
        """
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # Conv1d: вход (batch, 1, input_dim)
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout/2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout/2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # выход (batch, 256, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.conv_layers.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: (batch, input_dim)
        x = x.unsqueeze(1)  # (batch, 1, input_dim)
        x = self.conv_layers(x)  # (batch, 256, 1)
        x = x.squeeze(-1)  # (batch, 256)
        return self.classifier(x)