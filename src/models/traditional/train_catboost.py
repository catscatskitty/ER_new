#!/usr/bin/env python3
"""
Обучение CatBoost (акустика или акустика+фонетика)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import joblib
import json
from catboost import CatBoostClassifier
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
    parser = argparse.ArgumentParser(description='Обучение CatBoost')
    parser.add_argument('--config', type=str, default='configs', help='Путь к конфигам')
    parser.add_argument('--features', type=str, default='acoustic', 
                       choices=['acoustic', 'combined'],
                       help='Тип признаков: acoustic (38) или combined (38+27=65)')
    parser.add_argument('--force', action='store_true', help='Принудительное обучение')
    return parser.parse_args()


def load_features(data_dir, feature_type='acoustic'):
    """Загрузка признаков в зависимости от типа"""
    if feature_type == 'combined':
        X_train = np.load(data_dir / 'features_train_combined.npy')
        X_val = np.load(data_dir / 'features_val_combined.npy')
        X_test = np.load(data_dir / 'features_test_combined.npy')
    else:
        X_train = np.load(data_dir / 'features_train.npy')
        X_val = np.load(data_dir / 'features_val.npy')
        X_test = np.load(data_dir / 'features_test.npy')
    
    y_train = np.load(data_dir / 'labels_train.npy')
    y_val = np.load(data_dir / 'labels_val.npy')
    y_test = np.load(data_dir / 'labels_test.npy')
    
    return X_train, X_val, X_test, y_train, y_val, y_test


class CatBoostTrainer:
    def __init__(self, config_path='configs', feature_type='acoustic'):
        self.config_loader = ConfigLoader(config_path)
        self.training_config = self.config_loader.load_config('training_config')
        self.paths_config = self.config_loader.load_config('paths_config')
        self.file_manager = FileManager()
        self.feature_type = feature_type
        
        self.processed_root = Path(self.paths_config['paths']['processed_root'])
        self.models_dir = Path(self.paths_config['paths']['models_root']) / 'catboost'
        self.metrics_dir = Path(self.paths_config['paths']['metrics_root'])
        self.plots_dir = Path(self.paths_config['paths']['plots_root'])
        
        self.file_manager.ensure_dir(self.models_dir)
        self.file_manager.ensure_dir(self.metrics_dir)
        self.file_manager.ensure_dir(self.plots_dir)
        
        self.logger = setup_logger('catboost_trainer')
        
        training = self.training_config.get('training', {})
        set_random_seeds(training.get('random_seed', 42))
        
        self.logger.info(f"Тип признаков: {feature_type}")
    
    def run(self):
        self.logger.info("=" * 60)
        self.logger.info("ОБУЧЕНИЕ CATBOOST")
        self.logger.info(f"Тип признаков: {self.feature_type}")
        self.logger.info("=" * 60)
        
        if not self.processed_root.exists():
            self.logger.error(f"Директория с данными не найдена: {self.processed_root}")
            return None, None
        
        X_train, X_val, X_test, y_train, y_val, y_test = load_features(self.processed_root, self.feature_type)
        
        self.logger.info(f"Train: {X_train.shape}")
        self.logger.info(f"Val: {X_val.shape}")
        self.logger.info(f"Test: {X_test.shape}")
        
        # Нормализация
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Обучение
        model = CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_seed=42,
            verbose=False,
            early_stopping_rounds=10
        )
        
        model.fit(
            X_train_scaled, y_train,
            eval_set=(X_val_scaled, y_val),
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
        plt.title('Confusion Matrix - CatBoost')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'catboost_confusion_matrix.png')
        plt.close()
        
        # Важность признаков
        if hasattr(model, 'get_feature_importance'):
            importances = model.get_feature_importance()
            plt.figure(figsize=(10, 6))
            indices = np.argsort(importances)[::-1][:20]
            
            plt.bar(range(len(indices)), importances[indices])
            plt.title('CatBoost - Top 20 Feature Importances')
            plt.xlabel('Feature Index')
            plt.ylabel('Importance')
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'catboost_feature_importance.png')
            plt.close()
        
        # Сохранение
        model_path = self.models_dir / 'model.pkl'
        joblib.dump({'model': model, 'scaler': scaler}, model_path)
        self.logger.info(f"Модель сохранена в {model_path}")
        
        metrics = {
            'model': 'CatBoost',
            'feature_type': self.feature_type,
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
        
        with open(self.metrics_dir / 'catboost_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Метрики сохранены в {self.metrics_dir / 'catboost_metrics.json'}")
        
        return model, metrics


def main():
    args = parse_args()
    trainer = CatBoostTrainer(config_path=args.config, feature_type=args.features)
    trainer.run()


if __name__ == "__main__":
    main()