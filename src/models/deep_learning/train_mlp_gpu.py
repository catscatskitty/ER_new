#!/usr/bin/env python3
"""
Обучение многослойного перцептрона (MLP) на извлеченных признаках
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
from tqdm import tqdm
import json

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.utils.file_utils import FileManager
from src.utils.logger import setup_logger
from src.utils.config_loader import ConfigLoader
from src.utils.gpu_utils import setup_device, set_random_seeds, clear_gpu_memory, EarlyStopping, AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description='Обучение MLP на признаках')
    parser.add_argument('--config', type=str, default='configs', help='Путь к конфигам')
    parser.add_argument('--features', type=str, default='combined',
                       choices=['acoustic', 'phonetic', 'combined'],
                       help='Тип признаков для обучения')
    parser.add_argument('--force', action='store_true', help='Принудительное обучение')
    return parser.parse_args()


class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], num_classes=2, dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_features(data_dir, feature_type='combined'):
    # та же функция, что и ранее
    if feature_type == 'acoustic':
        X_train = np.load(data_dir / 'features_train.npy')
        X_val = np.load(data_dir / 'features_val.npy')
        X_test = np.load(data_dir / 'features_test.npy')
    elif feature_type == 'phonetic':
        X_train = np.load(data_dir / 'phonetic_train.npy')
        X_val = np.load(data_dir / 'phonetic_val.npy')
        X_test = np.load(data_dir / 'phonetic_test.npy')
    else:
        X_train_ac = np.load(data_dir / 'features_train.npy')
        X_train_ph = np.load(data_dir / 'phonetic_train.npy')
        X_train = np.hstack([X_train_ac, X_train_ph])

        X_val_ac = np.load(data_dir / 'features_val.npy')
        X_val_ph = np.load(data_dir / 'phonetic_val.npy')
        X_val = np.hstack([X_val_ac, X_val_ph])

        X_test_ac = np.load(data_dir / 'features_test.npy')
        X_test_ph = np.load(data_dir / 'phonetic_test.npy')
        X_test = np.hstack([X_test_ac, X_test_ph])

    y_train = np.load(data_dir / 'labels_train.npy')
    y_val = np.load(data_dir / 'labels_val.npy')
    y_test = np.load(data_dir / 'labels_test.npy')

    return X_train, X_val, X_test, y_train, y_val, y_test


class MLPTrainer:
    def __init__(self, config_path='configs'):
        self.config_loader = ConfigLoader(config_path)
        self.training_config = self.config_loader.load_config('training_config')
        self.paths_config = self.config_loader.load_config('paths_config')
        self.file_manager = FileManager()

        self.processed_root = Path(self.paths_config['paths']['processed_root'])
        self.models_dir = Path(self.paths_config['paths']['models_root']) / 'torch_models'
        self.metrics_dir = Path(self.paths_config['paths']['metrics_root'])
        self.plots_dir = Path(self.paths_config['paths']['plots_root'])

        self.file_manager.ensure_dir(self.models_dir)
        self.file_manager.ensure_dir(self.metrics_dir)
        self.file_manager.ensure_dir(self.plots_dir)

        self.logger = setup_logger('mlp_gpu_trainer')

        gpu_config = self.training_config['training'].get('gpu', {'enabled': True})
        self.device = setup_device(use_gpu=gpu_config.get('enabled', True), gpu_id=0)

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

    def run(self, feature_type='combined'):
        self.logger.info("=" * 60)
        self.logger.info(f"ОБУЧЕНИЕ MLP НА {feature_type.upper()} ПРИЗНАКАХ")
        self.logger.info("=" * 60)

        X_train, X_val, X_test, y_train, y_val, y_test = load_features(self.processed_root, feature_type)

        # Для MLP применяем нормализацию всегда (улучшает сходимость)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        self.logger.info("Применена нормализация StandardScaler")

        self.logger.info(f"Размерность признаков: {X_train.shape[1]}")

        class_weights = self.compute_class_weights(y_train)
        weights_tensor = torch.FloatTensor(class_weights).to(self.device)

        train_dataset = FeatureDataset(X_train, y_train)
        val_dataset = FeatureDataset(X_val, y_val)
        test_dataset = FeatureDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        input_dim = X_train.shape[1]
        model = MLPModel(input_dim=input_dim, num_classes=2).to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"Всего параметров: {total_params:,}")

        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        early_stopping = EarlyStopping(patience=10, min_delta=0.001)

        best_val_f1 = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer, epoch)
            val_loss, val_acc, val_f1, _, _ = self.validate(model, val_loader, criterion)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)

            scheduler.step(val_loss)

            self.logger.info(f"Эпоха {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'scaler': scaler
                }, self.models_dir / 'best_mlp.pth')
                self.logger.info(f"Сохранена лучшая модель (val_f1={val_f1:.4f})")

            if early_stopping(val_loss):
                self.logger.info(f"Early stopping на эпохе {epoch+1}")
                break

        checkpoint = torch.load(self.models_dir / 'best_mlp.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        scaler = checkpoint['scaler']

        # Тестирование
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
        plt.title('Confusion Matrix - MLP')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'mlp_confusion_matrix.png')
        plt.close()

        metrics = {
            'model': 'MLP',
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

        with open(self.metrics_dir / 'mlp_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        clear_gpu_memory()
        self.logger.info(f"\nMLP обучена. Accuracy: {report['accuracy']:.4f}")
        return model, metrics


def main():
    args = parse_args()
    trainer = MLPTrainer(config_path=args.config)
    trainer.run(feature_type=args.features)


if __name__ == "__main__":
    main()