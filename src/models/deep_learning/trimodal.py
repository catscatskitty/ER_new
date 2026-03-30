import torch
import torch.nn as nn


class TriModalModel(nn.Module):
    """Трёхмодальная модель: спектрограмма + MFCC + фонетика"""
    
    def __init__(self, n_mels=128, n_mfcc=13, phon_dim=27,
                 lstm_hidden=128, num_classes=2, dropout=0.3):
        super().__init__()
        
        # Спектрограмма -> CNN
        self.spec_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.spec_out = nn.Linear(64, 128)
        
        # MFCC -> LSTM
        self.mfcc_lstm = nn.LSTM(
            input_size=n_mfcc, hidden_size=lstm_hidden,
            num_layers=2, batch_first=True, bidirectional=True, dropout=dropout
        )
        self.mfcc_out = nn.Linear(lstm_hidden * 2, 128)
        
        # Фонетика -> MLP
        self.phon_mlp = nn.Sequential(
            nn.Linear(phon_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 128)
        )
        
        # Объединение
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, spec, mfcc, phon):
        # Спектрограмма
        spec_feat = self.spec_cnn(spec).view(spec.size(0), -1)
        spec_feat = nn.ReLU()(self.spec_out(spec_feat))
        
        # MFCC
        _, (hidden, _) = self.mfcc_lstm(mfcc)
        mfcc_feat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        mfcc_feat = nn.ReLU()(self.mfcc_out(mfcc_feat))
        
        # Фонетика
        phon_feat = self.phon_mlp(phon)
        
        # Объединение
        combined = torch.cat([spec_feat, mfcc_feat, phon_feat], dim=1)
        return self.classifier(combined)