#!/usr/bin/env python
"""
Обучение традиционных ML моделей на акустических признаках (38)
Использует готовые признаки из data/processed/acoustic/
"""

import numpy as np
import joblib
import yaml
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')


class TraditionalTrainer:
    def __init__(self, config_path='configs/training_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_root = Path('data/processed/acoustic')
        self.models_root = Path('results/trained_models')
        self.metrics_root = Path('results/metrics')
        
        self.models_root.mkdir(parents=True, exist_ok=True)
        self.metrics_root.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
    
    def _check_data_files(self):
        """Проверка наличия всех split файлов"""
        splits = ['train', 'val', 'test']
        for split in splits:
            feat_path = self.data_root / f'features_{split}.npy'
            label_path = self.data_root / f'labels_{split}.npy'
            if not feat_path.exists() or not label_path.exists():
                raise FileNotFoundError(
                    f"Missing {split} data. Run extract_all_features.py first."
                )
        print("✅ All data files found")
    
    def load_data(self):
        """Загрузка данных train/val/test"""
        print("\n📂 Loading data...")
        
        self.X_train = np.load(self.data_root / 'features_train.npy')
        self.y_train = np.load(self.data_root / 'labels_train.npy')
        self.X_val = np.load(self.data_root / 'features_val.npy')
        self.y_val = np.load(self.data_root / 'labels_val.npy')
        self.X_test = np.load(self.data_root / 'features_test.npy')
        self.y_test = np.load(self.data_root / 'labels_test.npy')
        
        print(f"  Train: {self.X_train.shape[0]} samples")
        print(f"  Val:   {self.X_val.shape[0]} samples")
        print(f"  Test:  {self.X_test.shape[0]} samples")
        print(f"  Features shape: {self.X_train.shape[1]}")
    
    def normalize_data(self):
        """Нормализация данных"""
        print("\n📊 Normalizing data...")
        
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"  Mean: {self.scaler.mean_[:5]}...")
        print(f"  Std:  {self.scaler.scale_[:5]}...")
    
    def train_logistic(self):
        """Обучение Logistic Regression"""
        print("\n🔹 Training Logistic Regression...")
        
        model = LogisticRegression(
            max_iter=self.config['traditional']['logistic']['max_iter'],
            random_state=42,
            n_jobs=-1
        )
        model.fit(self.X_train_scaled, self.y_train)
        
        # Оценка
        train_acc = model.score(self.X_train_scaled, self.y_train)
        val_acc = model.score(self.X_val_scaled, self.y_val)
        test_acc = model.score(self.X_test_scaled, self.y_test)
        
        y_pred = model.predict(self.X_test_scaled)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        cm = confusion_matrix(self.y_test, y_pred)
        
        print(f"  Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}, F1: {f1:.4f}")
        
        # Сохранение
        save_dir = self.models_root / 'logistic'
        save_dir.mkdir(exist_ok=True)
        joblib.dump({'model': model, 'scaler': self.scaler}, save_dir / 'model.pkl')
        
        self.results['logistic'] = {
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'test_accuracy': float(test_acc),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist()
        }
        
        return model
    
    def train_random_forest(self):
        """Обучение Random Forest"""
        print("\n🔹 Training Random Forest...")
        
        params = self.config['traditional']['random_forest']
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            random_state=42,
            n_jobs=-1
        )
        model.fit(self.X_train_scaled, self.y_train)
        
        train_acc = model.score(self.X_train_scaled, self.y_train)
        val_acc = model.score(self.X_val_scaled, self.y_val)
        test_acc = model.score(self.X_test_scaled, self.y_test)
        
        y_pred = model.predict(self.X_test_scaled)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        cm = confusion_matrix(self.y_test, y_pred)
        
        print(f"  Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}, F1: {f1:.4f}")
        
        save_dir = self.models_root / 'random_forest'
        save_dir.mkdir(exist_ok=True)
        joblib.dump({'model': model, 'scaler': self.scaler}, save_dir / 'model.pkl')
        
        self.results['random_forest'] = {
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'test_accuracy': float(test_acc),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'feature_importance': model.feature_importances_.tolist()
        }
        
        return model
    
    def train_xgboost(self):
        """Обучение XGBoost"""
        print("\n🔹 Training XGBoost...")
        
        params = self.config['traditional']['xgboost']
        model = xgb.XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        model.fit(self.X_train_scaled, self.y_train)
        
        train_acc = model.score(self.X_train_scaled, self.y_train)
        val_acc = model.score(self.X_val_scaled, self.y_val)
        test_acc = model.score(self.X_test_scaled, self.y_test)
        
        y_pred = model.predict(self.X_test_scaled)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        cm = confusion_matrix(self.y_test, y_pred)
        
        print(f"  Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}, F1: {f1:.4f}")
        
        save_dir = self.models_root / 'xgboost'
        save_dir.mkdir(exist_ok=True)
        joblib.dump({'model': model, 'scaler': self.scaler}, save_dir / 'model.pkl')
        
        self.results['xgboost'] = {
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'test_accuracy': float(test_acc),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'feature_importance': model.feature_importances_.tolist()
        }
        
        return model
    
    def train_catboost(self):
        """Обучение CatBoost"""
        print("\n🔹 Training CatBoost...")
        
        params = self.config['traditional']['catboost']
        model = CatBoostClassifier(
            iterations=params['iterations'],
            learning_rate=params['learning_rate'],
            depth=params['depth'],
            random_seed=42,
            verbose=False,
            thread_count=-1
        )
        model.fit(self.X_train_scaled, self.y_train)
        
        train_acc = model.score(self.X_train_scaled, self.y_train)
        val_acc = model.score(self.X_val_scaled, self.y_val)
        test_acc = model.score(self.X_test_scaled, self.y_test)
        
        y_pred = model.predict(self.X_test_scaled)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        cm = confusion_matrix(self.y_test, y_pred)
        
        print(f"  Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}, F1: {f1:.4f}")
        
        save_dir = self.models_root / 'catboost'
        save_dir.mkdir(exist_ok=True)
        joblib.dump({'model': model, 'scaler': self.scaler}, save_dir / 'model.pkl')
        
        self.results['catboost'] = {
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'test_accuracy': float(test_acc),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'feature_importance': model.feature_importances_.tolist()
        }
        
        return model
    
    def save_results(self):
        """Сохранение метрик"""
        print("\n💾 Saving results...")
        
        # JSON
        with open(self.metrics_root / 'traditional_metrics.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # CSV
        df = pd.DataFrame([
            {
                'Model': name,
                'Train_Accuracy': metrics['train_accuracy'],
                'Val_Accuracy': metrics['val_accuracy'],
                'Test_Accuracy': metrics['test_accuracy'],
                'F1_Score': metrics['f1_score']
            }
            for name, metrics in self.results.items()
        ])
        df.to_csv(self.metrics_root / 'traditional_metrics.csv', index=False)
        
        print(f"  Saved to {self.metrics_root}/traditional_metrics.json and .csv")
    
    def run(self):
        """Запуск полного обучения"""
        print("="*60)
        print("Traditional Models Trainer")
        print("="*60)
        
        self._check_data_files()
        self.load_data()
        self.normalize_data()
        
        self.train_logistic()
        self.train_random_forest()
        self.train_xgboost()
        self.train_catboost()
        
        self.save_results()
        
        print("\n✅ Traditional models training completed!")
        
        # Вывод лучшей модели
        best = max(self.results.items(), key=lambda x: x[1]['test_accuracy'])
        print(f"\n🏆 Best model: {best[0]} with test accuracy = {best[1]['test_accuracy']:.4f}")
        
        return self.results


if __name__ == "__main__":
    import pandas as pd
    trainer = TraditionalTrainer()
    trainer.run()