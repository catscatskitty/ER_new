#!/usr/bin/env python
"""
Сравнение всех обученных моделей
Использует готовые признаки из data/processed/ и обученные модели из results/trained_models/
"""

import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Импорт моделей
from src.models.deep_learning.cnn_spectrogram import CNN2D
from src.models.deep_learning.lstm_mfcc import LSTM_MFCC
from src.models.deep_learning.hybrid_spectrogram import HybridSpectrogram
from src.models.deep_learning.trimodal import TriModalModel


class ModelEvaluator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_root = Path('results/trained_models')
        self.data_root = Path('data/processed')
        self.results = {}
        
        # Проверка наличия тестовых данных
        self._check_test_data()
    
    def _check_test_data(self):
        """Проверка наличия тестовых данных"""
        required = [
            'acoustic/features_test.npy', 'acoustic/labels_test.npy',
            'spectrograms/spectrograms_test.npy', 'spectrograms/labels_test.npy',
            'mfcc_sequences/mfcc_test.npy', 'mfcc_sequences/labels_test.npy'
        ]
        for path in required:
            if not (self.data_root / path).exists():
                raise FileNotFoundError(f"Missing test data: {path}. Run extract_all_features.py first.")
        print("✅ Test data found")
    
    def evaluate_traditional_models(self):
        """Оценка традиционных ML моделей"""
        print("\n📊 Evaluating Traditional Models...")
        
        X_test = np.load(self.data_root / 'acoustic' / 'features_test.npy')
        y_test = np.load(self.data_root / 'acoustic' / 'labels_test.npy')
        
        models = ['logistic', 'random_forest', 'xgboost', 'catboost']
        
        for model_name in models:
            model_path = self.models_root / model_name / 'model.pkl'
            if not model_path.exists():
                print(f"  ⚠️ Model {model_name} not found, skipping")
                continue
            
            data = joblib.load(model_path)
            model = data['model']
            scaler = data['scaler']
            
            X_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_scaled)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            cm = confusion_matrix(y_test, y_pred)
            
            self.results[model_name] = {
                'type': 'traditional',
                'accuracy': float(acc),
                'f1_weighted': float(f1),
                'confusion_matrix': cm.tolist(),
                'classification_report': classification_report(y_test, y_pred, target_names=['Human', 'Robot'])
            }
            
            print(f"  {model_name}: acc={acc:.4f}, f1={f1:.4f}")
    
    def evaluate_cnn(self):
        """Оценка 2D CNN на спектрограммах"""
        print("\n📊 Evaluating CNN on Spectrograms...")
        
        X_test = np.load(self.data_root / 'spectrograms' / 'spectrograms_test.npy')
        y_test = np.load(self.data_root / 'spectrograms' / 'labels_test.npy')
        
        # Нормализация
        norm_path = self.models_root / 'torch_models' / 'cnn_normalization.npy'
        if not norm_path.exists():
            print("  ⚠️ CNN normalization file not found, skipping")
            return
        
        norm_data = np.load(norm_path, allow_pickle=True).item()
        mean, std = norm_data['mean'], norm_data['std']
        X_test = (X_test - mean) / (std + 1e-8)
        X_test = X_test[:, np.newaxis, :, :]
        
        # Загрузка модели
        model_path = self.models_root / 'torch_models' / 'best_cnn_spectrogram.pth'
        if not model_path.exists():
            print("  ⚠️ CNN model not found, skipping")
            return
        
        model = CNN2D().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        # Инференс
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        with torch.no_grad():
            outputs = model(X_tensor)
            _, y_pred = torch.max(outputs, 1)
        
        y_pred = y_pred.cpu().numpy()
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        
        self.results['cnn_spectrogram'] = {
            'type': 'deep',
            'accuracy': float(acc),
            'f1_weighted': float(f1),
            'confusion_matrix': cm.tolist()
        }
        
        print(f"  cnn_spectrogram: acc={acc:.4f}, f1={f1:.4f}")
    
    def evaluate_lstm(self):
        """Оценка LSTM на MFCC"""
        print("\n📊 Evaluating LSTM on MFCC...")
        
        X_test = np.load(self.data_root / 'mfcc_sequences' / 'mfcc_test.npy')
        y_test = np.load(self.data_root / 'mfcc_sequences' / 'labels_test.npy')
        
        model_path = self.models_root / 'torch_models' / 'best_lstm_mfcc.pth'
        if not model_path.exists():
            print("  ⚠️ LSTM model not found, skipping")
            return
        
        model = LSTM_MFCC(input_dim=X_test.shape[2]).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        with torch.no_grad():
            outputs = model(X_tensor)
            _, y_pred = torch.max(outputs, 1)
        
        y_pred = y_pred.cpu().numpy()
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        
        self.results['lstm_mfcc'] = {
            'type': 'deep',
            'accuracy': float(acc),
            'f1_weighted': float(f1),
            'confusion_matrix': cm.tolist()
        }
        
        print(f"  lstm_mfcc: acc={acc:.4f}, f1={f1:.4f}")
    
    def evaluate_hybrid(self):
        """Оценка гибридной модели на спектрограммах"""
        print("\n📊 Evaluating Hybrid CNN+LSTM on Spectrograms...")
        
        X_test = np.load(self.data_root / 'spectrograms' / 'spectrograms_test.npy')
        y_test = np.load(self.data_root / 'spectrograms' / 'labels_test.npy')
        
        norm_path = self.models_root / 'torch_models' / 'hybrid_normalization.npy'
        if not norm_path.exists():
            print("  ⚠️ Hybrid normalization file not found, skipping")
            return
        
        norm_data = np.load(norm_path, allow_pickle=True).item()
        mean, std = norm_data['mean'], norm_data['std']
        X_test = (X_test - mean) / (std + 1e-8)
        X_test = X_test[:, np.newaxis, :, :]
        
        model_path = self.models_root / 'torch_models' / 'best_hybrid_spectrogram.pth'
        if not model_path.exists():
            print("  ⚠️ Hybrid model not found, skipping")
            return
        
        model = HybridSpectrogram().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        with torch.no_grad():
            outputs = model(X_tensor)
            _, y_pred = torch.max(outputs, 1)
        
        y_pred = y_pred.cpu().numpy()
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        
        self.results['hybrid_spectrogram'] = {
            'type': 'deep',
            'accuracy': float(acc),
            'f1_weighted': float(f1),
            'confusion_matrix': cm.tolist()
        }
        
        print(f"  hybrid_spectrogram: acc={acc:.4f}, f1={f1:.4f}")
    
    def generate_comparison_table(self):
        """Создание таблицы сравнения"""
        print("\n" + "="*60)
        print("📊 Comparison Results")
        print("="*60)
        
        df = pd.DataFrame([
            {
                'Model': name,
                'Accuracy': info['accuracy'],
                'F1_Weighted': info['f1_weighted'],
                'Type': info['type']
            }
            for name, info in self.results.items()
        ]).sort_values('Accuracy', ascending=False)
        
        print(df.to_string(index=False))
        
        # Сохранение
        df.to_csv(self.models_root / 'comparison_results.csv', index=False)
        
        with open(self.models_root / 'comparison_results.json', 'w') as f:
            json.dump({k: v for k, v in self.results.items() if k != 'classification_report'}, 
                     f, indent=2)
        
        # Визуализация
        plt.figure(figsize=(12, 6))
        bars = plt.bar(df['Model'], df['Accuracy'], color=['#2E86AB' if t == 'deep' else '#A23B72' for t in df['Type']])
        plt.axhline(y=0.5, color='r', linestyle='--', label='Random (0.5)')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title('Model Comparison on Test Set')
        plt.xticks(rotation=45, ha='right')
        plt.legend(['Random Baseline', 'Deep Learning', 'Traditional ML'])
        plt.tight_layout()
        plt.savefig(self.models_root / 'comparison_plot.png', dpi=150)
        plt.close()
        
        print(f"\n✅ Results saved to {self.models_root}/comparison_results.csv")
        print(f"✅ Plot saved to {self.models_root}/comparison_plot.png")
    
    def run_all(self):
        """Запуск полной оценки"""
        print("="*60)
        print("Model Evaluation")
        print("="*60)
        
        self.evaluate_traditional_models()
        self.evaluate_cnn()
        self.evaluate_lstm()
        self.evaluate_hybrid()
        
        self.generate_comparison_table()
        
        print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run_all()