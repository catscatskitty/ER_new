#!/usr/bin/env python
"""
Обучение всех моделей (традиционные + нейросетевые)
Использует готовые признаки из data/processed/
"""
import argparse
import os
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в PYTHONPATH
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.environ['PYTHONPATH'] = str(project_root) + os.pathsep + os.environ.get('PYTHONPATH', '')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml
import json
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
import joblib

from src.models.deep_learning.cnn_spectrogram import CNN2D
from src.models.deep_learning.lstm_mfcc import LSTM_MFCC
from src.models.deep_learning.hybrid_spectrogram import HybridSpectrogram
from src.models.deep_learning.trimodal import TriModalModel


class TrainingPipeline:
    def __init__(self, config_path='configs/training_config.yaml'):
        # Полный путь к конфигу
        config_full_path = project_root / config_path
        with open(config_full_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.data_root = project_root / 'data' / 'processed'
        self.models_root = project_root / 'results' / 'trained_models'
        self.models_root.mkdir(parents=True, exist_ok=True)
        (self.models_root / 'torch_models').mkdir(exist_ok=True)
        (self.models_root / 'trimodal').mkdir(exist_ok=True)
    
    def _check_split_files(self):
        """Проверка наличия всех split файлов"""
        splits = ['train', 'val', 'test']
        for split in splits:
            # Проверяем акустические признаки (для традиционных моделей)
            acoustic_path = self.data_root / 'acoustic' / f'features_{split}.npy'
            if not acoustic_path.exists():
                print(f"⚠️ {acoustic_path} not found, skipping...")
                continue
            
            # Проверяем спектрограммы (для CNN)
            spec_path = self.data_root / 'spectrograms' / f'spectrograms_{split}.npy'
            if not spec_path.exists():
                print(f"⚠️ {spec_path} not found, CNN will be skipped...")
            
            # Проверяем MFCC последовательности (для LSTM)
            mfcc_path = self.data_root / 'mfcc_sequences' / f'mfcc_{split}.npy'
            if not mfcc_path.exists():
                print(f"⚠️ {mfcc_path} not found, LSTM will be skipped...")
        
        print("✅ Split files check completed")
    
    def train_traditional(self):
        """Обучение традиционных ML моделей на акустических признаках (38)"""
        print("\n" + "="*60)
        print("Training Traditional ML Models")
        print("="*60)
        
        # Проверяем наличие данных
        acoustic_dir = self.data_root / 'acoustic'
        if not (acoustic_dir / 'features_train.npy').exists():
            print("⚠️ Acoustic features not found. Skipping traditional models.")
            return {}
        
        # Загрузка данных (готовые split файлы)
        X_train = np.load(acoustic_dir / 'features_train.npy')
        y_train = np.load(acoustic_dir / 'labels_train.npy')
        X_val = np.load(acoustic_dir / 'features_val.npy')
        y_val = np.load(acoustic_dir / 'labels_val.npy')
        X_test = np.load(acoustic_dir / 'features_test.npy')
        y_test = np.load(acoustic_dir / 'labels_test.npy')
        
        print(f"Train: {X_train.shape[0]} samples")
        print(f"Val:   {X_val.shape[0]} samples")
        print(f"Test:  {X_test.shape[0]} samples")
        
        # Нормализация
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='logloss'),
            'catboost': CatBoostClassifier(iterations=100, random_seed=42, verbose=False, thread_count=-1)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_scaled, y_train)
            
            train_acc = model.score(X_train_scaled, y_train)
            val_acc = model.score(X_val_scaled, y_val)
            test_acc = model.score(X_test_scaled, y_test)
            
            results[name] = {
                'train_accuracy': float(train_acc),
                'val_accuracy': float(val_acc),
                'test_accuracy': float(test_acc)
            }
            
            # Сохранение модели и scaler
            save_dir = self.models_root / name
            save_dir.mkdir(exist_ok=True)
            joblib.dump({'model': model, 'scaler': scaler}, save_dir / 'model.pkl')
            
            print(f"  Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
        
        # Сохранение результатов
        with open(self.models_root / 'traditional_metrics.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def train_cnn(self):
        """Обучение 2D CNN на спектрограммах"""
        print("\n" + "="*60)
        print("Training 2D CNN on Spectrograms")
        print("="*60)
        
        # Проверяем наличие данных
        spec_dir = self.data_root / 'spectrograms'
        if not (spec_dir / 'spectrograms_train.npy').exists():
            print("⚠️ Spectrogram features not found. Skipping CNN.")
            return None
        
        # Загрузка данных
        X_train = np.load(spec_dir / 'spectrograms_train.npy')
        y_train = np.load(spec_dir / 'labels_train.npy')
        X_val = np.load(spec_dir / 'spectrograms_val.npy')
        y_val = np.load(spec_dir / 'labels_val.npy')
        
        print(f"Train: {X_train.shape[0]} samples, shape: {X_train.shape}")
        print(f"Val:   {X_val.shape[0]} samples, shape: {X_val.shape}")
        
        # Нормализация
        mean, std = X_train.mean(), X_train.std()
        X_train = (X_train - mean) / (std + 1e-8)
        X_val = (X_val - mean) / (std + 1e-8)
        
        # Добавление канального измерения
        X_train = X_train[:, np.newaxis, :, :]
        X_val = X_val[:, np.newaxis, :, :]
        
        # DataLoader
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
            batch_size=self.config['cnn']['batch_size'], shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
            batch_size=self.config['cnn']['batch_size'], shuffle=False
        )
        
        # Модель
        model = CNN2D().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config['cnn']['lr'])
        
        best_val_acc = 0
        for epoch in range(self.config['cnn']['epochs']):
            model.train()
            train_loss = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Валидация
            model.eval()
            correct = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    pred = torch.argmax(model(x), dim=1)
                    correct += (pred == y).sum().item()
            val_acc = correct / len(y_val)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), self.models_root / 'torch_models' / 'best_cnn_spectrogram.pth')
                # Сохраняем параметры нормализации
                norm_params = {'mean': mean, 'std': std}
                np.save(self.models_root / 'torch_models' / 'cnn_normalization.npy', norm_params)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Val Acc={val_acc:.4f}")
        
        print(f"Best val accuracy: {best_val_acc:.4f}")
        return best_val_acc
    
    def train_lstm(self):
        """Обучение LSTM на MFCC-последовательностях"""
        print("\n" + "="*60)
        print("Training LSTM on MFCC Sequences")
        print("="*60)
        
        # Проверяем наличие данных
        mfcc_dir = self.data_root / 'mfcc_sequences'
        if not (mfcc_dir / 'mfcc_train.npy').exists():
            print("⚠️ MFCC features not found. Skipping LSTM.")
            return None
        
        # Загрузка данных
        X_train = np.load(mfcc_dir / 'mfcc_train.npy')
        y_train = np.load(mfcc_dir / 'labels_train.npy')
        X_val = np.load(mfcc_dir / 'mfcc_val.npy')
        y_val = np.load(mfcc_dir / 'labels_val.npy')
        
        print(f"Train: {X_train.shape[0]} samples, shape: {X_train.shape}")
        print(f"Val:   {X_val.shape[0]} samples, shape: {X_val.shape}")
        
        # DataLoader
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
            batch_size=self.config['lstm']['batch_size'], shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
            batch_size=self.config['lstm']['batch_size'], shuffle=False
        )
        
        # Модель
        model = LSTM_MFCC(input_dim=X_train.shape[2]).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config['lstm']['lr'])
        
        best_val_acc = 0
        for epoch in range(self.config['lstm']['epochs']):
            model.train()
            train_loss = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Валидация
            model.eval()
            correct = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    pred = torch.argmax(model(x), dim=1)
                    correct += (pred == y).sum().item()
            val_acc = correct / len(y_val)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), self.models_root / 'torch_models' / 'best_lstm_mfcc.pth')
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Val Acc={val_acc:.4f}")
        
        print(f"Best val accuracy: {best_val_acc:.4f}")
        return best_val_acc
    
    def train_hybrid(self):
        """Обучение гибридной модели на спектрограммах"""
        print("\n" + "="*60)
        print("Training Hybrid CNN+LSTM on Spectrograms")
        print("="*60)
        
        # Проверяем наличие данных
        spec_dir = self.data_root / 'spectrograms'
        if not (spec_dir / 'spectrograms_train.npy').exists():
            print("⚠️ Spectrogram features not found. Skipping Hybrid model.")
            return None
        
        # Загрузка данных
        X_train = np.load(spec_dir / 'spectrograms_train.npy')
        y_train = np.load(spec_dir / 'labels_train.npy')
        X_val = np.load(spec_dir / 'spectrograms_val.npy')
        y_val = np.load(spec_dir / 'labels_val.npy')
        
        # Нормализация
        mean, std = X_train.mean(), X_train.std()
        X_train = (X_train - mean) / (std + 1e-8)
        X_val = (X_val - mean) / (std + 1e-8)
        
        X_train = X_train[:, np.newaxis, :, :]
        X_val = X_val[:, np.newaxis, :, :]
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
            batch_size=self.config['hybrid']['batch_size'], shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
            batch_size=self.config['hybrid']['batch_size'], shuffle=False
        )
        
        model = HybridSpectrogram().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config['hybrid']['lr'])
        
        best_val_acc = 0
        for epoch in range(self.config['hybrid']['epochs']):
            model.train()
            train_loss = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            model.eval()
            correct = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    pred = torch.argmax(model(x), dim=1)
                    correct += (pred == y).sum().item()
            val_acc = correct / len(y_val)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), self.models_root / 'torch_models' / 'best_hybrid_spectrogram.pth')
                norm_params = {'mean': mean, 'std': std}
                np.save(self.models_root / 'torch_models' / 'hybrid_normalization.npy', norm_params)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Val Acc={val_acc:.4f}")
        
        print(f"Best val accuracy: {best_val_acc:.4f}")
        return best_val_acc
    
    def run_all(self):
        """Запуск обучения всех моделей"""
        print("="*60)
        print("Starting Training Pipeline")
        print("="*60)
        
        # Проверка наличия split файлов
        self._check_split_files()
        
        # Обучение
        self.train_traditional()
        self.train_cnn()
        self.train_lstm()
        self.train_hybrid()
        
        print("\n" + "="*60)
        print("✅ All models trained successfully!")
        print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train selected models')
    parser.add_argument('--models', nargs='+', 
                       choices=['traditional', 'cnn', 'lstm', 'hybrid', 'trimodal', 'all'],
                       default=['all'],
                       help='Models to train')
    args = parser.parse_args()
    
    pipeline = TrainingPipeline()
    
    if 'all' in args.models:
        pipeline.run_all()
    else:
        if 'traditional' in args.models:
            pipeline.train_traditional()
        if 'cnn' in args.models:
            pipeline.train_cnn()
        if 'lstm' in args.models:
            pipeline.train_lstm()
        if 'hybrid' in args.models:
            pipeline.train_hybrid()
        if 'trimodal' in args.models:
            pipeline.train_trimodal()