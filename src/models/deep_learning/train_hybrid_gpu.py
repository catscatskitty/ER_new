#!/usr/bin/env python3
"""
Обучение гибридной CNN-LSTM модели на извлеченных признаках
Полная версия с поддержкой --config
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import time

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.utils.file_utils import FileManager
from src.utils.logger import setup_logger
from src.utils.config_loader import ConfigLoader
from src.utils.gpu_utils import (
    setup_device, set_random_seeds, clear_gpu_memory,
    get_optimal_batch_size, EarlyStopping, AverageMeter
)


def parse_args():
    parser = argparse.ArgumentParser(description='Обучение Hybrid модели на признаках')
    parser.add_argument('--config', type=str, default='configs', help='Путь к конфигам')
    parser.add_argument('--features', type=str, default='acoustic', 
                       choices=['acoustic', 'linguistic', 'combined'],
                       help='Тип признаков для обучения')
    parser.add_argument('--force', action='store_true', help='Принудительное обучение')
    return parser.parse_args()


class HybridFeatureModel(nn.Module):
    """Гибридная CNN-LSTM модель для работы с извлеченными признаками"""
    
    def __init__(self, input_dim=38, hidden_size=64, num_classes=2, dropout=0.3):
        super().__init__()
        
        # CNN часть
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # LSTM часть
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
    
    def forward(self, x):
        # x shape: (batch, features)
        x = x.unsqueeze(1)  # (batch, 1, features)
        
        # CNN
        x = self.cnn(x)  # (batch, 64, 1)
        x = x.squeeze(-1)  # (batch, 64)
        x = x.unsqueeze(1)  # (batch, 1, 64) для LSTM
        
        # LSTM
        lstm_out, (hidden, _) = self.lstm(x)
        hidden_concat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        
        # Классификация
        out = self.classifier(hidden_concat)
        return out


class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_features(data_dir, feature_type='acoustic'):
    """Загрузка признаков в зависимости от типа"""
    if feature_type == 'acoustic':
        X_train = np.load(data_dir / 'features_train.npy')
        X_val = np.load(data_dir / 'features_val.npy')
        X_test = np.load(data_dir / 'features_test.npy')
    elif feature_type == 'linguistic':
        X_train = np.load(data_dir / 'linguistic_train.npy')
        X_val = np.load(data_dir / 'linguistic_val.npy')
        X_test = np.load(data_dir / 'linguistic_test.npy')
    else:  # combined
        X_train_ac = np.load(data_dir / 'features_train.npy')
        X_train_lin = np.load(data_dir / 'linguistic_train.npy')
        X_train = np.hstack([X_train_ac, X_train_lin])
        
        X_val_ac = np.load(data_dir / 'features_val.npy')
        X_val_lin = np.load(data_dir / 'linguistic_val.npy')
        X_val = np.hstack([X_val_ac, X_val_lin])
        
        X_test_ac = np.load(data_dir / 'features_test.npy')
        X_test_lin = np.load(data_dir / 'linguistic_test.npy')
        X_test = np.hstack([X_test_ac, X_test_lin])
    
    y_train = np.load(data_dir / 'labels_train.npy')
    y_val = np.load(data_dir / 'labels_val.npy')
    y_test = np.load(data_dir / 'labels_test.npy')
    
    return X_train, X_val, X_test, y_train, y_val, y_test


class HybridTrainer:
# В начале класса CNNTrainer, замените строки с загрузкой конфигов:

    def __init__(self, config_path='configs'):
        self.config_loader = ConfigLoader(config_path)
        
        # Загружаем конфиги с проверкой
        self.training_config = self.config_loader.load_config('training_config')
        self.paths_config = self.config_loader.load_config('paths_config')
        
        # Проверка на пустые конфиги
        if not self.paths_config:
            self.paths_config = {'paths': {
                'processed_root': './data/processed',
                'models_root': './results/trained_models',
                'metrics_root': './results/metrics',
                'plots_root': './results/plots'
            }}
        
        if not self.training_config:
            self.training_config = {'training': {
                'random_seed': 42,
                'gpu': {'enabled': True},
                'deep_learning': {
                    'batch_size': 64,
                    'epochs': 50,
                    'learning_rate': 0.001
                }
            }}
        
        self.file_manager = FileManager()
        
        # Загружаем пути
        paths = self.paths_config.get('paths', {})
        self.processed_root = Path(paths.get('processed_root', './data/processed'))
        self.models_dir = Path(paths.get('models_root', './results/trained_models')) / 'torch_models'
        self.metrics_dir = Path(paths.get('metrics_root', './results/metrics'))
        self.plots_dir = Path(paths.get('plots_root', './results/plots'))
        
        self.file_manager.ensure_dir(self.models_dir)
        self.file_manager.ensure_dir(self.metrics_dir)
        self.file_manager.ensure_dir(self.plots_dir)
        
        self.logger = setup_logger('hybrid_gpu_trainer')
        
        # Настройка устройства
        training = self.training_config.get('training', {})
        gpu_config = training.get('gpu', {'enabled': True})
        self.device = setup_device(use_gpu=gpu_config.get('enabled', True), gpu_id=0)
        
        # Устанавливаем seed
        set_random_seeds(training.get('random_seed', 42))
        
        # Параметры обучения
        dl_config = training.get('deep_learning', {})
        self.batch_size = dl_config.get('batch_size', 64)
        self.epochs = dl_config.get('epochs', 50)
        self.learning_rate = dl_config.get('learning_rate', 0.001)
        
        self.logger.info(f"Устройство: {self.device}")
        self.logger.info(f"Batch size: {self.batch_size}")
    
    def compute_class_weights(self, y_train):
        classes = np.array([0, 1])
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        
        self.logger.info(f"\nВеса классов: Human={class_weights[0]:.3f}, Robot={class_weights[1]:.3f}")
        return class_weights
    
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        model.train()
        running_loss = AverageMeter()
        running_acc = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} Train')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            acc = (predicted == labels).sum().item() / labels.size(0)
            
            running_loss.update(loss.item(), labels.size(0))
            running_acc.update(acc, labels.size(0))
            
            pbar.set_postfix({'loss': f'{running_loss.avg:.4f}', 'acc': f'{running_acc.avg*100:.2f}%'})
        
        return running_loss.avg, running_acc.avg
    
    def validate(self, model, val_loader, criterion):
        model.eval()
        running_loss = AverageMeter()
        running_acc = AverageMeter()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                acc = (predicted == labels).sum().item() / labels.size(0)
                
                running_loss.update(loss.item(), labels.size(0))
                running_acc.update(acc, labels.size(0))
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = running_loss.avg
        val_acc = running_acc.avg
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return val_loss, val_acc, val_f1, all_preds, all_labels
    
    def run(self, feature_type='acoustic'):
        self.logger.info("=" * 60)
        self.logger.info("ОБУЧЕНИЕ ГИБРИДНОЙ CNN-LSTM НА ПРИЗНАКАХ")
        self.logger.info(f"Тип признаков: {feature_type}")
        self.logger.info("=" * 60)
        
        X_train, X_val, X_test, y_train, y_val, y_test = load_features(self.processed_root, feature_type)
        
        class_weights = self.compute_class_weights(y_train)
        weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        
        train_dataset = FeatureDataset(X_train, y_train)
        val_dataset = FeatureDataset(X_val, y_val)
        test_dataset = FeatureDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        input_dim = X_train.shape[1]
        model = HybridFeatureModel(input_dim=input_dim, num_classes=2).to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"Всего параметров: {total_params:,}")
        
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        
        best_val_f1 = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
        
        for epoch in range(self.epochs):
            self.logger.info(f"\nЭпоха {epoch+1}/{self.epochs}")
            
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer, epoch)
            val_loss, val_acc, val_f1, _, _ = self.validate(model, val_loader, criterion)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            
            scheduler.step(val_loss)
            
            self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
            self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, Val F1: {val_f1:.4f}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), self.models_dir / 'best_hybrid.pth')
                self.logger.info(f"Сохранена лучшая модель (val_f1={val_f1:.4f})")
            
            if early_stopping(val_loss):
                self.logger.info(f"Early stopping на эпохе {epoch+1}")
                break
        
        model.load_state_dict(torch.load(self.models_dir / 'best_hybrid.pth'))
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("ТЕСТИРОВАНИЕ")
        self.logger.info("=" * 60)
        
        _, _, _, all_preds, all_labels = self.validate(model, test_loader, criterion)
        
        report = classification_report(all_labels, all_preds, target_names=['human', 'robot'], output_dict=True)
        self.logger.info("\n" + classification_report(all_labels, all_preds, target_names=['human', 'robot']))
        
        robot_f1 = f1_score(all_labels, all_preds, pos_label=1)
        self.logger.info(f"F1-score для роботов: {robot_f1:.4f}")
        
        cm = confusion_matrix(all_labels, all_preds)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['human', 'robot'],
                   yticklabels=['human', 'robot'])
        plt.title('Confusion Matrix - Hybrid CNN-LSTM')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'hybrid_confusion_matrix.png')
        plt.close()
        
        # Графики
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history['train_loss'], label='Train')
        axes[0].plot(history['val_loss'], label='Validation')
        axes[0].set_title('Loss')
        axes[0].legend()
        
        axes[1].plot([acc*100 for acc in history['train_acc']], label='Train')
        axes[1].plot([acc*100 for acc in history['val_acc']], label='Validation')
        axes[1].set_title('Accuracy (%)')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'hybrid_training_history.png')
        plt.close()
        
        metrics = {
            'model': 'Hybrid',
            'accuracy': report['accuracy'],
            'precision_human': report['human']['precision'],
            'recall_human': report['human']['recall'],
            'f1_human': report['human']['f1-score'],
            'precision_robot': report['robot']['precision'],
            'recall_robot': report['robot']['recall'],
            'f1_robot': report['robot']['f1-score'],
            'robot_f1': robot_f1,
            'confusion_matrix': cm.tolist()
        }
        
        with open(self.metrics_dir / 'hybrid_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        clear_gpu_memory()
        
        self.logger.info(f"\nГибридная модель обучена. Accuracy: {report['accuracy']:.4f}")
        return model, metrics


def main():
    args = parse_args()
    trainer = HybridTrainer(config_path=args.config)
    trainer.run(feature_type=args.features)


if __name__ == "__main__":
    main()