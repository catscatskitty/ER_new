#!/usr/bin/env python3
"""
Трёхмодальная модель: CNN (спектрограммы) + LSTM (MFCC) + MLP (фонетика)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
    parser = argparse.ArgumentParser(description='Трёхмодальная модель')
    parser.add_argument('--config', type=str, default='configs', help='Путь к конфигам')
    parser.add_argument('--force', action='store_true', help='Принудительное обучение')
    return parser.parse_args()


class CNNBranch(nn.Module):
    """CNN для спектрограмм"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2)
        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.global_avg(x)
        x = x.view(x.size(0), -1)  # (batch, 128)
        return x


class LSTMBranch(nn.Module):
    """LSTM для MFCC-последовательностей"""
    def __init__(self, input_size=13, hidden_size=128, num_layers=2, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.3
        )
        self.num_directions = 2 if bidirectional else 1

    def forward(self, x):
        lstm_out, (hidden, _) = self.lstm(x)
        if self.num_directions == 2:
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            hidden_concat = torch.cat((hidden_forward, hidden_backward), dim=1)
        else:
            hidden_concat = hidden[-1, :, :]
        return hidden_concat  # (batch, hidden_size * 2)


class MLPBranch(nn.Module):
    """MLP для фонетических признаков"""
    def __init__(self, input_dim=27, hidden_dims=[64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hdim
        self.net = nn.Sequential(*layers)
        self.out_dim = prev_dim

    def forward(self, x):
        return self.net(x)


class TriModalModel(nn.Module):
    def __init__(self, cnn_out=128, lstm_out=256, mlp_out=32, num_classes=2, dropout=0.5):
        super().__init__()
        self.cnn_branch = CNNBranch()
        self.lstm_branch = LSTMBranch()
        self.mlp_branch = MLPBranch(hidden_dims=[64, mlp_out])

        fusion_dim = cnn_out + lstm_out + mlp_out
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, spec, mfcc, phonetic):
        cnn_feat = self.cnn_branch(spec)
        lstm_feat = self.lstm_branch(mfcc)
        mlp_feat = self.mlp_branch(phonetic)
        combined = torch.cat([cnn_feat, lstm_feat, mlp_feat], dim=1)
        return self.fusion(combined)


class TriModalDataset(Dataset):
    def __init__(self, spectrograms, mfcc_seq, phonetic, labels):
        self.spectrograms = torch.FloatTensor(spectrograms).unsqueeze(1)  # (batch, 1, 128, 128)
        self.mfcc_seq = torch.FloatTensor(mfcc_seq)                       # (batch, time, 13)
        self.phonetic = torch.FloatTensor(phonetic)                       # (batch, 27)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.spectrograms[idx], self.mfcc_seq[idx], self.phonetic[idx], self.labels[idx]


def load_data(processed_root):
    spec_dir = processed_root / 'spectrograms'
    mfcc_dir = processed_root / 'mfcc_sequences'
    phonetic_dir = processed_root

    X_spec_train = np.load(spec_dir / 'spectrograms_train.npy')
    X_spec_val = np.load(spec_dir / 'spectrograms_val.npy')
    X_spec_test = np.load(spec_dir / 'spectrograms_test.npy')

    X_mfcc_train = np.load(mfcc_dir / 'mfcc_train.npy')
    X_mfcc_val = np.load(mfcc_dir / 'mfcc_val.npy')
    X_mfcc_test = np.load(mfcc_dir / 'mfcc_test.npy')

    X_phon_train = np.load(phonetic_dir / 'phonetic_train.npy')
    X_phon_val = np.load(phonetic_dir / 'phonetic_val.npy')
    X_phon_test = np.load(phonetic_dir / 'phonetic_test.npy')

    y_train = np.load(spec_dir / 'labels_train.npy')
    y_val = np.load(spec_dir / 'labels_val.npy')
    y_test = np.load(spec_dir / 'labels_test.npy')

    return (X_spec_train, X_spec_val, X_spec_test,
            X_mfcc_train, X_mfcc_val, X_mfcc_test,
            X_phon_train, X_phon_val, X_phon_test,
            y_train, y_val, y_test)


class TriModalTrainer:
    def __init__(self, config_path='configs'):
        self.config_loader = ConfigLoader(config_path)
        self.training_config = self.config_loader.load_config('training_config')
        self.paths_config = self.config_loader.load_config('paths_config')
        self.file_manager = FileManager()

        self.processed_root = Path(self.paths_config['paths']['processed_root'])
        self.models_dir = Path(self.paths_config['paths']['models_root']) / 'trimodal'
        self.metrics_dir = Path(self.paths_config['paths']['metrics_root'])
        self.plots_dir = Path(self.paths_config['paths']['plots_root'])

        self.file_manager.ensure_dir(self.models_dir)
        self.file_manager.ensure_dir(self.metrics_dir)
        self.file_manager.ensure_dir(self.plots_dir)

        self.logger = setup_logger('trimodal')

        gpu_config = self.training_config['training'].get('gpu', {'enabled': True})
        self.device = setup_device(use_gpu=gpu_config.get('enabled', True), gpu_id=0)

        set_random_seeds(self.training_config['training']['random_seed'])

        dl_config = self.training_config['training']['deep_learning']
        self.batch_size = dl_config.get('batch_size', 64)
        self.epochs = dl_config.get('epochs', 50)
        self.learning_rate = dl_config.get('learning_rate', 0.001)

    def load_data(self):
        return load_data(self.processed_root)

    def compute_class_weights(self, y_train):
        classes = np.array([0, 1])
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        self.logger.info(f"Веса классов: Human={weights[0]:.3f}, Robot={weights[1]:.3f}")
        return weights

    def run(self):
        self.logger.info("=" * 60)
        self.logger.info("ОБУЧЕНИЕ ТРЁХМОДАЛЬНОЙ МОДЕЛИ (спектрограммы + MFCC + фонетика)")
        self.logger.info("=" * 60)

        (X_spec_train, X_spec_val, X_spec_test,
         X_mfcc_train, X_mfcc_val, X_mfcc_test,
         X_phon_train, X_phon_val, X_phon_test,
         y_train, y_val, y_test) = self.load_data()

        self.logger.info(f"Спектрограммы train: {X_spec_train.shape}")
        self.logger.info(f"MFCC train: {X_mfcc_train.shape}")
        self.logger.info(f"Фонетика train: {X_phon_train.shape}")

        class_weights = self.compute_class_weights(y_train)
        weights_tensor = torch.FloatTensor(class_weights).to(self.device)

        train_dataset = TriModalDataset(X_spec_train, X_mfcc_train, X_phon_train, y_train)
        val_dataset = TriModalDataset(X_spec_val, X_mfcc_val, X_phon_val, y_val)
        test_dataset = TriModalDataset(X_spec_test, X_mfcc_test, X_phon_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        model = TriModalModel().to(self.device)
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
            for spec, mfcc, phon, labels in pbar:
                spec = spec.to(self.device)
                mfcc = mfcc.to(self.device)
                phon = phon.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(spec, mfcc, phon)
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
                for spec, mfcc, phon, labels in val_loader:
                    spec = spec.to(self.device)
                    mfcc = mfcc.to(self.device)
                    phon = phon.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(spec, mfcc, phon)
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
                torch.save(model.state_dict(), self.models_dir / 'best_trimodal.pth')
                self.logger.info(f"Сохранена лучшая модель (val_acc={best_val_acc:.4f})")

            if early_stopping(val_loss.avg):
                self.logger.info(f"Early stopping на эпохе {epoch+1}")
                break

        # Тестирование
        model.load_state_dict(torch.load(self.models_dir / 'best_trimodal.pth'))
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for spec, mfcc, phon, labels in test_loader:
                spec = spec.to(self.device)
                mfcc = mfcc.to(self.device)
                phon = phon.to(self.device)
                labels = labels.to(self.device)
                outputs = model(spec, mfcc, phon)
                _, pred = torch.max(outputs, 1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        report = classification_report(all_labels, all_preds, target_names=['human', 'robot'])
        self.logger.info("\n" + report)

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['human','robot'], yticklabels=['human','robot'])
        plt.title('Confusion Matrix - TriModal (CNN+LSTM+MLP)')
        plt.savefig(self.plots_dir / 'trimodal_cm.png')
        plt.close()

        metrics = {
            'model': 'TriModal',
            'accuracy': report['accuracy'],
            'confusion_matrix': cm.tolist()
        }
        with open(self.metrics_dir / 'trimodal_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        clear_gpu_memory()
        self.logger.info(f"Обучение завершено. Accuracy: {report['accuracy']:.4f}")


def main():
    args = parse_args()
    trainer = TriModalTrainer(config_path=args.config)
    trainer.run()


if __name__ == "__main__":
    main()