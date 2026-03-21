#!/usr/bin/env python3
"""
Обучение XGBoost
Адаптировано для разных версий библиотеки
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import joblib
import json
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.utils.file_utils import FileManager
from src.utils.logger import setup_logger
from src.utils.config_loader import ConfigLoader
from src.utils.gpu_utils import set_random_seeds


def parse_args():
    parser = argparse.ArgumentParser(description='Обучение XGBoost')
    parser.add_argument('--config', type=str, default='configs', help='Путь к конфигам')
    parser.add_argument('--features', type=str, default='acoustic', 
                       choices=['acoustic', 'phonetic', 'combined'],
                       help='Тип признаков для обучения')
    parser.add_argument('--force', action='store_true', help='Принудительное обучение')
    return parser.parse_args()


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
        X_train_ac = np.load(data_dir / 'features_train.npy')
        X_train_ph = np.load(data_dir / 'phonetic_train.npy')
        X_val_ac = np.load(data_dir / 'features_val.npy')
        X_val_ph = np.load(data_dir / 'phonetic_val.npy')
        X_test_ac = np.load(data_dir / 'features_test.npy')
        X_test_ph = np.load(data_dir / 'phonetic_test.npy')
        
        # Обрезаем до минимальной длины
        min_train = min(X_train_ac.shape[0], X_train_ph.shape[0])
        min_val = min(X_val_ac.shape[0], X_val_ph.shape[0])
        min_test = min(X_test_ac.shape[0], X_test_ph.shape[0])
        
        if X_train_ac.shape[0] != X_train_ph.shape[0]:
            print(f" Обрезаю train: акустика {X_train_ac.shape[0]}, фонетика {X_train_ph.shape[0]}  {min_train}")
            X_train_ac = X_train_ac[:min_train]
            X_train_ph = X_train_ph[:min_train]
        if X_val_ac.shape[0] != X_val_ph.shape[0]:
            print(f" Обрезаю val: акустика {X_val_ac.shape[0]}, фонетика {X_val_ph.shape[0]}  {min_val}")
            X_val_ac = X_val_ac[:min_val]
            X_val_ph = X_val_ph[:min_val]
        if X_test_ac.shape[0] != X_test_ph.shape[0]:
            print(f" Обрезаю test: акустика {X_test_ac.shape[0]}, фонетика {X_test_ph.shape[0]}  {min_test}")
            X_test_ac = X_test_ac[:min_test]
            X_test_ph = X_test_ph[:min_test]
        
        X_train = np.hstack([X_train_ac, X_train_ph])
        X_val = np.hstack([X_val_ac, X_val_ph])
        X_test = np.hstack([X_test_ac, X_test_ph])
    
    y_train = np.load(data_dir / 'labels_train.npy')
    y_val = np.load(data_dir / 'labels_val.npy')
    y_test = np.load(data_dir / 'labels_test.npy')
    
    # Обрезаем метки до совпадающей длины
    if X_train.shape[0] != y_train.shape[0]:
        y_train = y_train[:X_train.shape[0]]
    if X_val.shape[0] != y_val.shape[0]:
        y_val = y_val[:X_val.shape[0]]
    if X_test.shape[0] != y_test.shape[0]:
        y_test = y_test[:X_test.shape[0]]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


class XGBoostTrainer:
    def __init__(self, config_path='configs'):
        self.config_loader = ConfigLoader(config_path)
        self.training_config = self.config_loader.load_config('training_config')
        self.paths_config = self.config_loader.load_config('paths_config')
        self.file_manager = FileManager()
        
        self.processed_root = Path(self.paths_config['paths']['processed_root'])
        self.models_dir = Path(self.paths_config['paths']['models_root']) / 'xgboost'
        self.metrics_dir = Path(self.paths_config['paths']['metrics_root'])
        self.plots_dir = Path(self.paths_config['paths']['plots_root'])
        
        self.file_manager.ensure_dir(self.models_dir)
        self.file_manager.ensure_dir(self.metrics_dir)
        self.file_manager.ensure_dir(self.plots_dir)
        
        self.logger = setup_logger('xgboost_trainer')
        
        training = self.training_config.get('training', {})
        set_random_seeds(training.get('random_seed', 42))
    
    def run(self, feature_type='acoustic'):
        self.logger.info("=" * 60)
        self.logger.info("ОБУЧЕНИЕ XGBOOST")
        self.logger.info(f"Тип признаков: {feature_type}")
        self.logger.info("=" * 60)
        
        if not self.processed_root.exists():
            self.logger.error(f"❌ Директория с данными не найдена: {self.processed_root}")
            return None, None
        
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = load_features(self.processed_root, feature_type)
        except FileNotFoundError as e:
            self.logger.error(f"❌ Файлы признаков не найдены: {e}")
            return None, None
        
        self.logger.info(f"Train: {X_train.shape}")
        self.logger.info(f"Val: {X_val.shape}")
        self.logger.info(f"Test: {X_test.shape}")
        
        # Нормализация
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Создаём модель
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        # Обучение с early stopping (универсальный способ)
        self.logger.info("Обучение модели...")
        try:
            # Пробуем передать early_stopping_rounds в fit (новая версия)
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
        except TypeError:
            # Если не сработало, пробуем через конструктор (старая версия)
            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss',
                early_stopping_rounds=10
            )
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
        
        # Валидация
        val_pred = model.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average='weighted')
        self.logger.info(f"Validation Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Тестирование
        test_pred = model.predict(X_test_scaled)
        
        report = classification_report(y_test, test_pred, target_names=['human', 'robot'], output_dict=True)
        self.logger.info("\n" + classification_report(y_test, test_pred, target_names=['human', 'robot']))
        
        robot_f1 = f1_score(y_test, test_pred, pos_label=1)
        self.logger.info(f"F1-score для роботов: {robot_f1:.4f}")
        
        cm = confusion_matrix(y_test, test_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['human', 'robot'],
                   yticklabels=['human', 'robot'])
        plt.title('Confusion Matrix - XGBoost')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'xgboost_confusion_matrix.png')
        plt.close()
        
        # Сохранение
        model_path = self.models_dir / 'model.pkl'
        joblib.dump({'model': model, 'scaler': scaler}, model_path)
        self.logger.info(f"✅ Модель сохранена в {model_path}")
        
        metrics = {
            'model': 'XGBoost',
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
        
        metrics_path = self.metrics_dir / 'xgboost_metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"✅ Метрики сохранены в {metrics_path}")
        
        return model, metrics


def main():
    args = parse_args()
    trainer = XGBoostTrainer(config_path=args.config)
    trainer.run(feature_type=args.features)


if __name__ == "__main__":
    main()