#!/usr/bin/env python3
"""
Обучение 2D CNN на спектрограммах
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
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
    parser = argparse.ArgumentParser(description='Обучение CNN на спектрограммах')
    parser.add_argument('--config', type=str, default='configs', help='Путь к конфигам')
    parser.add_argument('--force', action='store_true', help='Принудительное обучение')
    return parser.parse_args()


class SpectrogramDataset(Dataset):
    def __init__(self, X, y):
        # X: (n_samples, n_mels, time_steps)
        self.X = torch.FloatTensor(X).unsqueeze(1)  # добавляем канал (1, height, width)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleCNN2D(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)

        # После трёх пулингов размер уменьшится в 8 раз
        # После адаптивного пула сведём к 1x1
        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.global_avg(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SpectrogramTrainer:
    def __init__(self, config_path='configs'):
        self.config_loader = ConfigLoader(config_path)
        self.training_config = self.config_loader.load_config('training_config')
        self.paths_config = self.config_loader.load_config('paths_config')
        self.file_manager = FileManager()

        self.spectrogram_dir = Path(self.paths_config['paths']['processed_root']) / 'spectrograms'
        self.models_dir = Path(self.paths_config['paths']['models_root']) / 'spectrogram_cnn'
        self.metrics_dir = Path(self.paths_config['paths']['metrics_root'])
        self.plots_dir = Path(self.paths_config['paths']['plots_root'])

        self.file_manager.ensure_dir(self.models_dir)
        self.file_manager.ensure_dir(self.metrics_dir)
        self.file_manager.ensure_dir(self.plots_dir)

        self.logger = setup_logger('spectrogram_cnn')

        gpu_config = self.training_config['training'].get('gpu', {'enabled': True})
        self.device = setup_device(use_gpu=gpu_config.get('enabled', True), gpu_id=0)

        set_random_seeds(self.training_config['training']['random_seed'])

        dl_config = self.training_config['training']['deep_learning']
        self.batch_size = dl_config.get('batch_size', 64)
        self.epochs = dl_config.get('epochs', 50)
        self.learning_rate = dl_config.get('learning_rate', 0.001)

    def load_data(self):
        X_train = np.load(self.spectrogram_dir / 'spectrograms_train.npy')
        X_val = np.load(self.spectrogram_dir / 'spectrograms_val.npy')
        X_test = np.load(self.spectrogram_dir / 'spectrograms_test.npy')
        y_train = np.load(self.spectrogram_dir / 'labels_train.npy')
        y_val = np.load(self.spectrogram_dir / 'labels_val.npy')
        y_test = np.load(self.spectrogram_dir / 'labels_test.npy')
        return X_train, X_val, X_test, y_train, y_val, y_test

    def compute_class_weights(self, y_train):
        classes = np.array([0, 1])
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        self.logger.info(f"Веса классов: Human={weights[0]:.3f}, Robot={weights[1]:.3f}")
        return weights

    def run(self):
        self.logger.info("=" * 60)
        self.logger.info("ОБУЧЕНИЕ 2D CNN НА СПЕКТРОГРАММАХ")
        self.logger.info("=" * 60)

        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()
        self.logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        class_weights = self.compute_class_weights(y_train)
        weights_tensor = torch.FloatTensor(class_weights).to(self.device)

        train_dataset = SpectrogramDataset(X_train, y_train)
        val_dataset = SpectrogramDataset(X_val, y_val)
        test_dataset = SpectrogramDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        model = SimpleCNN2D(num_classes=2).to(self.device)
        self.logger.info(f"Всего параметров: {sum(p.numel() for p in model.parameters()):,}")

        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        early_stopping = EarlyStopping(patience=10)

        best_val_acc = 0
        for epoch in range(self.epochs):
            model.train()
            running_loss = AverageMeter()
            running_acc = AverageMeter()
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                _, pred = torch.max(outputs, 1)
                acc = (pred == labels).sum().item() / labels.size(0)
                running_loss.update(loss.item(), labels.size(0))
                running_acc.update(acc, labels.size(0))
                pbar.set_postfix({'loss': running_loss.avg, 'acc': running_acc.avg})

            # Валидация
            model.eval()
            val_loss = AverageMeter()
            val_acc = AverageMeter()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, pred = torch.max(outputs, 1)
                    val_loss.update(loss.item(), labels.size(0))
                    val_acc.update((pred == labels).sum().item() / labels.size(0), labels.size(0))
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            val_f1 = f1_score(all_labels, all_preds, average='weighted')
            self.logger.info(f"Epoch {epoch+1}: train_loss={running_loss.avg:.4f}, train_acc={running_acc.avg:.4f}, "
                             f"val_loss={val_loss.avg:.4f}, val_acc={val_acc.avg:.4f}, val_f1={val_f1:.4f}")

            scheduler.step(val_loss.avg)

            if val_acc.avg > best_val_acc:
                best_val_acc = val_acc.avg
                torch.save(model.state_dict(), self.models_dir / 'best_spectrogram_cnn.pth')
                self.logger.info(f"Сохранена лучшая модель (val_acc={best_val_acc:.4f})")

            if early_stopping(val_loss.avg):
                self.logger.info(f"Early stopping на эпохе {epoch+1}")
                break

        # Тестирование
        model.load_state_dict(torch.load(self.models_dir / 'best_spectrogram_cnn.pth'))
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, pred = torch.max(outputs, 1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        report = classification_report(all_labels, all_preds, target_names=['human', 'robot'])
        self.logger.info("\n" + report)

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['human','robot'], yticklabels=['human','robot'])
        plt.title('Confusion Matrix - Spectrogram CNN')
        plt.savefig(self.plots_dir / 'spectrogram_cnn_cm.png')
        plt.close()

        # Сохранение метрик
        metrics = {
            'model': 'SpectrogramCNN',
            'accuracy': report['accuracy'],
            'confusion_matrix': cm.tolist()
        }
        with open(self.metrics_dir / 'spectrogram_cnn_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        clear_gpu_memory()
        self.logger.info(f"Обучение завершено. Accuracy: {report['accuracy']:.4f}")


def main():
    args = parse_args()
    trainer = SpectrogramTrainer(config_path=args.config)
    trainer.run()


if __name__ == "__main__":
    main()