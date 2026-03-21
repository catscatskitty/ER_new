#!/usr/bin/env python3
"""
Обучение CNN модели на извлеченных признаках
Использует класс CNNAudioClassifier из cnn_model.py
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import json

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.utils.file_utils import FileManager
from src.utils.logger import setup_logger
from src.utils.config_loader import ConfigLoader
from src.utils.gpu_utils import setup_device, set_random_seeds, clear_gpu_memory, EarlyStopping, AverageMeter
from src.models.deep_learning.cnn_model import CNNAudioClassifier


def parse_args():
    parser = argparse.ArgumentParser(description='Обучение CNN на признаках')
    parser.add_argument('--config', type=str, default='configs', help='Путь к конфигам')
    parser.add_argument('--features', type=str, default='acoustic', 
                       choices=['acoustic', 'phonetic', 'combined'],
                       help='Тип признаков для обучения')
    parser.add_argument('--force', action='store_true', help='Принудительное обучение')
    return parser.parse_args()


class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_features(data_dir, feature_type='acoustic'):
    """Загрузка признаков в зависимости от типа с проверкой размерностей"""
    data_dir = Path(data_dir)
    
    if feature_type == 'acoustic':
        X_train = np.load(data_dir / 'features_train.npy')
        X_val = np.load(data_dir / 'features_val.npy')
        X_test = np.load(data_dir / 'features_test.npy')
    elif feature_type == 'phonetic':
        X_train = np.load(data_dir / 'phonetic_train.npy')
        X_val = np.load(data_dir / 'phonetic_val.npy')
        X_test = np.load(data_dir / 'phonetic_test.npy')
    else:  # combined
        # Загружаем акустические и фонетические
        X_train_ac = np.load(data_dir / 'features_train.npy')
        X_train_ph = np.load(data_dir / 'phonetic_train.npy')
        X_val_ac = np.load(data_dir / 'features_val.npy')
        X_val_ph = np.load(data_dir / 'phonetic_val.npy')
        X_test_ac = np.load(data_dir / 'features_test.npy')
        X_test_ph = np.load(data_dir / 'phonetic_test.npy')
        
        # Проверяем размерности и обрезаем до минимальной
        min_train = min(X_train_ac.shape[0], X_train_ph.shape[0])
        min_val = min(X_val_ac.shape[0], X_val_ph.shape[0])
        min_test = min(X_test_ac.shape[0], X_test_ph.shape[0])
        
        if X_train_ac.shape[0] != X_train_ph.shape[0]:
            print(f" Несоответствие train: акустика {X_train_ac.shape[0]}, фонетика {X_train_ph.shape[0]}. Обрезаю до {min_train}")
            X_train_ac = X_train_ac[:min_train]
            X_train_ph = X_train_ph[:min_train]
        
        if X_val_ac.shape[0] != X_val_ph.shape[0]:
            print(f" Несоответствие val: акустика {X_val_ac.shape[0]}, фонетика {X_val_ph.shape[0]}. Обрезаю до {min_val}")
            X_val_ac = X_val_ac[:min_val]
            X_val_ph = X_val_ph[:min_val]
        
        if X_test_ac.shape[0] != X_test_ph.shape[0]:
            print(f" Несоответствие test: акустика {X_test_ac.shape[0]}, фонетика {X_test_ph.shape[0]}. Обрезаю до {min_test}")
            X_test_ac = X_test_ac[:min_test]
            X_test_ph = X_test_ph[:min_test]
        
        # Объединяем
        X_train = np.hstack([X_train_ac, X_train_ph])
        X_val = np.hstack([X_val_ac, X_val_ph])
        X_test = np.hstack([X_test_ac, X_test_ph])
    
    # Загружаем метки
    y_train = np.load(data_dir / 'labels_train.npy')
    y_val = np.load(data_dir / 'labels_val.npy')
    y_test = np.load(data_dir / 'labels_test.npy')
    
    # Обрезаем метки до совпадающей размерности
    if X_train.shape[0] != y_train.shape[0]:
        print(f" Обрезаю метки train: {y_train.shape[0]}  {X_train.shape[0]}")
        y_train = y_train[:X_train.shape[0]]
    if X_val.shape[0] != y_val.shape[0]:
        print(f" Обрезаю метки val: {y_val.shape[0]}  {X_val.shape[0]}")
        y_val = y_val[:X_val.shape[0]]
    if X_test.shape[0] != y_test.shape[0]:
        print(f" Обрезаю метки test: {y_test.shape[0]}  {X_test.shape[0]}")
        y_test = y_test[:X_test.shape[0]]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


class CNNTrainer:
    def __init__(self, config_path='configs'):
        self.config_loader = ConfigLoader(config_path)
        self.training_config = self.config_loader.load_config('training_config')
        self.paths_config = self.config_loader.load_config('paths_config')
        self.file_manager = FileManager()
        
        self.processed_root = Path(self.paths_config['paths']['processed_root'])
        self.models_dir = Path(self.paths_config['paths']['models_root']) / 'cnn_gpu'
        self.metrics_dir = Path(self.paths_config['paths']['metrics_root'])
        self.plots_dir = Path(self.paths_config['paths']['plots_root'])
        
        self.file_manager.ensure_dir(self.models_dir)
        self.file_manager.ensure_dir(self.metrics_dir)
        self.file_manager.ensure_dir(self.plots_dir)
        
        self.logger = setup_logger('cnn_gpu_trainer')
        self.device = setup_device(use_gpu=True)
        
        set_random_seeds(self.training_config['training']['random_seed'])
        
        dl_config = self.training_config['training']['deep_learning']
        self.batch_size = dl_config['batch_size']
        self.epochs = dl_config['epochs']
        self.learning_rate = dl_config['learning_rate']
        
        self.logger.info(f"Устройство: {self.device}")
        self.logger.info(f"Batch size: {self.batch_size}")
    
    def compute_class_weights(self, y_train):
        classes = np.array([0, 1])
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        self.logger.info(f"Веса классов: Human={class_weights[0]:.3f}, Robot={class_weights[1]:.3f}")
        return class_weights
    
    def train_epoch(self, model, loader, criterion, optimizer):
        model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        for inputs, labels in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, pred = torch.max(outputs, 1)
            acc = (pred == labels).sum().item() / labels.size(0)
            loss_meter.update(loss.item(), labels.size(0))
            acc_meter.update(acc, labels.size(0))
        
        return loss_meter.avg, acc_meter.avg
    
    def validate(self, model, loader, criterion):
        model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, pred = torch.max(outputs, 1)
                acc = (pred == labels).sum().item() / labels.size(0)
                
                loss_meter.update(loss.item(), labels.size(0))
                acc_meter.update(acc, labels.size(0))
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return loss_meter.avg, acc_meter.avg, all_preds, all_labels
    
    def run(self, feature_type='acoustic'):
        self.logger.info("=" * 60)
        self.logger.info(f"ОБУЧЕНИЕ CNN НА {feature_type.upper()} ПРИЗНАКАХ")
        self.logger.info("=" * 60)
        
        # Загружаем данные
        X_train, X_val, X_test, y_train, y_val, y_test = load_features(self.processed_root, feature_type)
        
        self.logger.info(f"Размерность признаков: {X_train.shape[1]}")
        self.logger.info(f"Train: {X_train.shape}")
        self.logger.info(f"Val: {X_val.shape}")
        self.logger.info(f"Test: {X_test.shape}")
        
        # Нормализация
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        import joblib
        joblib.dump(scaler, self.models_dir / 'scaler.pkl')
        
        # Веса классов
        class_weights = self.compute_class_weights(y_train)
        weights = torch.FloatTensor(class_weights).to(self.device)
        
        # Датасеты
        train_dataset = FeatureDataset(X_train, y_train)
        val_dataset = FeatureDataset(X_val, y_val)
        test_dataset = FeatureDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Модель
        input_dim = X_train.shape[1]
        model = CNNAudioClassifier(input_dim=input_dim).to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"Всего параметров: {total_params:,}")
        
        # Функция потерь и оптимизатор
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        early_stopping = EarlyStopping(patience=10)
        
        best_val_f1 = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc, val_preds, val_labels = self.validate(model, val_loader, criterion)
            val_f1 = f1_score(val_labels, val_preds, average='weighted')
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            scheduler.step(val_loss)
            
            self.logger.info(f"Эпоха {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), self.models_dir / 'best_cnn.pth')
                self.logger.info(f"Сохранена лучшая модель (val_f1={val_f1:.4f})")
            
            if early_stopping(val_loss):
                self.logger.info(f"Early stopping на эпохе {epoch+1}")
                break
        
        # Тестирование
        model.load_state_dict(torch.load(self.models_dir / 'best_cnn.pth'))
        _, _, all_preds, all_labels = self.validate(model, test_loader, criterion)
        
        report = classification_report(all_labels, all_preds, target_names=['human', 'robot'], output_dict=True)
        self.logger.info("\n" + classification_report(all_labels, all_preds, target_names=['human', 'robot']))
        
        robot_f1 = f1_score(all_labels, all_preds, pos_label=1)
        self.logger.info(f"F1-score для роботов: {robot_f1:.4f}")
        
        cm = confusion_matrix(all_labels, all_preds)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['human', 'robot'],
                   yticklabels=['human', 'robot'])
        plt.title(f'Confusion Matrix - CNN ({feature_type})')
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'cnn_{feature_type}_cm.png')
        plt.close()
        
        metrics = {
            'model': f'CNN_{feature_type}',
            'accuracy': report['accuracy'],
            'precision_human': report['human']['precision'],
            'recall_human': report['human']['recall'],
            'f1_human': report['human']['f1-score'],
            'precision_robot': report['robot']['precision'],
            'recall_robot': report['robot']['recall'],
            'f1_robot': report['robot']['f1-score'],
            'robot_f1': robot_f1,
            'confusion_matrix': cm.tolist(),
            'feature_type': feature_type,
            'input_dim': input_dim
        }
        
        with open(self.metrics_dir / f'cnn_{feature_type}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        clear_gpu_memory()
        
        return model, metrics


def main():
    args = parse_args()
    trainer = CNNTrainer(config_path=args.config)
    trainer.run(feature_type=args.features)


if __name__ == "__main__":
    main()