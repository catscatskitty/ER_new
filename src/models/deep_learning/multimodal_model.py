import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalModel(nn.Module):
    """
    Мультимодальная модель (акустика + лингвистика)
    """
    
    def __init__(self, acoustic_dim: int = 38, linguistic_dim: int = 80,
                 fusion_dim: int = 128, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        
        # Акустическая ветка
        self.acoustic_encoder = nn.Sequential(
            nn.Linear(acoustic_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Лингвистическая ветка
        self.linguistic_encoder = nn.Sequential(
            nn.Linear(linguistic_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(128, fusion_dim),  # 64 + 64 = 128
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Классификатор
        self.classifier = nn.Linear(64, num_classes)
        
    def forward(self, acoustic_features, linguistic_features):
        # Кодирование каждой модальности
        acoustic_encoded = self.acoustic_encoder(acoustic_features)
        linguistic_encoded = self.linguistic_encoder(linguistic_features)
        
        # Объединение
        fused = torch.cat([acoustic_encoded, linguistic_encoded], dim=1)
        
        # Fusion
        fused = self.fusion(fused)
        
        # Классификация
        output = self.classifier(fused)
        
        return output