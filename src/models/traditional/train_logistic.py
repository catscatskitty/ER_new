#!/usr/bin/env python3
"""
Обучение логистической регрессии
Полная версия с поддержкой --config
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import joblib
import json
from sklearn.linear_model import LogisticRegression
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
    parser = argparse.ArgumentParser(description='Обучение Logistic Regression')
    parser.add_argument('--config', type=str, default='configs', help='Путь к конфигам')
    parser.add_argument('--features', type=str, default='acoustic', 
                       choices=['acoustic', 'linguistic', 'combined'],
                       help='Тип признаков для обучения')
    parser.add_argument('--force', action='store_true', help='Принудительное обучение')
    return parser.parse_args()


def load_features(data_dir, feature_type='acoustic'):
    """Загрузка признаков - всегда 38"""
    
    # Загружаем акустические признаки (они должны быть 38)
    X_train = np.load(data_dir / 'features_train.npy')
    X_val = np.load(data_dir / 'features_val.npy')
    X_test = np.load(data_dir / 'features_test.npy')
    
    # Проверяем размерность
    if X_train.shape[1] != 38:
        print(f"Внимание: признаки имеют размерность {X_train.shape[1]}, ожидалось 38")
        # Обрезаем или дополняем до 38
        if X_train.shape[1] > 38:
            X_train = X_train[:, :38]
            X_val = X_val[:, :38]
            X_test = X_test[:, :38]
        elif X_train.shape[1] < 38:
            # Дополняем нулями
            pad_width = ((0, 0), (0, 38 - X_train.shape[1]))
            X_train = np.pad(X_train, pad_width, mode='constant')
            X_val = np.pad(X_val, pad_width, mode='constant')
            X_test = np.pad(X_test, pad_width, mode='constant')
    
    y_train = np.load(data_dir / 'labels_train.npy')
    y_val = np.load(data_dir / 'labels_val.npy')
    y_test = np.load(data_dir / 'labels_test.npy')
    
    return X_train, X_val, X_test, y_train, y_val, y_test


class LogisticTrainer:
    def __init__(self, config_path='configs'):
        # Загружаем конфиги с обработкой ошибок
        self.config_loader = ConfigLoader(config_path)
        
        # Пытаемся загрузить конфиги, если не получается - используем значения по умолчанию
        try:
            self.paths_config = self.config_loader.load_config('paths_config')
        except Exception as e:
            print(f"⚠️ Ошибка загрузки paths_config: {e}")
            self.paths_config = {'paths': {}}
        
        try:
            self.training_config = self.config_loader.load_config('training_config')
        except Exception as e:
            print(f"⚠️ Ошибка загрузки training_config: {e}")
            self.training_config = {'training': {}}
        
        self.file_manager = FileManager()
        
        # Загружаем пути с значениями по умолчанию
        paths = self.paths_config.get('paths', {})
        
        # Определяем корневые директории
        self.processed_root = Path(paths.get('processed_root', 'data/processed'))
        models_root = Path(paths.get('models_root', 'results/trained_models'))
        self.models_dir = models_root / 'logistic'
        self.metrics_dir = Path(paths.get('metrics_root', 'results/metrics'))
        self.plots_dir = Path(paths.get('plots_root', 'results/plots'))
        
        # Создаем директории
        self.file_manager.ensure_dir(self.models_dir)
        self.file_manager.ensure_dir(self.metrics_dir)
        self.file_manager.ensure_dir(self.plots_dir)
        
        self.logger = setup_logger('logistic_trainer')
        
        # Устанавливаем seed
        training = self.training_config.get('training', {})
        set_random_seeds(training.get('random_seed', 42))
    
    def run(self, feature_type='acoustic'):
        self.logger.info("=" * 60)
        self.logger.info("ОБУЧЕНИЕ LOGISTIC REGRESSION")
        self.logger.info(f"Тип признаков: {feature_type}")
        self.logger.info("=" * 60)
        
        # Проверяем существование директории с данными
        if not self.processed_root.exists():
            self.logger.error(f"❌ Директория с данными не найдена: {self.processed_root}")
            self.logger.error("Сначала запустите извлечение признаков")
            return None, None
        
        # Загружаем данные
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = load_features(self.processed_root, feature_type)
        except FileNotFoundError as e:
            self.logger.error(f"❌ Файлы признаков не найдены: {e}")
            self.logger.error("Сначала запустите build_feature_pipeline.py")
            return None, None
        
        self.logger.info(f"Train: {X_train.shape}, Human: {np.sum(y_train==0)}, Robot: {np.sum(y_train==1)}")
        self.logger.info(f"Val: {X_val.shape}, Human: {np.sum(y_val==0)}, Robot: {np.sum(y_val==1)}")
        self.logger.info(f"Test: {X_test.shape}, Human: {np.sum(y_test==0)}, Robot: {np.sum(y_test==1)}")
        
        # Нормализация
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Обучение
        self.logger.info("Обучение модели...")
        model = LogisticRegression(
            C=1.0, 
            max_iter=1000, 
            class_weight='balanced', 
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Валидация
        val_pred = model.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average='weighted')
        self.logger.info(f"Validation Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Тестирование
        test_pred = model.predict(X_test_scaled)
        test_proba = model.predict_proba(X_test_scaled)
        
        # Отчет классификации
        report = classification_report(y_test, test_pred, target_names=['human', 'robot'], output_dict=True)
        self.logger.info("\n" + classification_report(y_test, test_pred, target_names=['human', 'robot']))
        
        robot_f1 = f1_score(y_test, test_pred, pos_label=1)
        self.logger.info(f"F1-score для роботов: {robot_f1:.4f}")
        
        # Матрица ошибок
        cm = confusion_matrix(y_test, test_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['human', 'robot'],
                   yticklabels=['human', 'robot'])
        plt.title('Confusion Matrix - Logistic Regression')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'logistic_confusion_matrix.png')
        plt.close()
        
        # Сохранение
        model_path = self.models_dir / 'model.pkl'
        joblib.dump({'model': model, 'scaler': scaler}, model_path)
        self.logger.info(f"✅ Модель сохранена в {model_path}")
        
        # Метрики
        metrics = {
            'model': 'Logistic Regression',
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
        
        metrics_path = self.metrics_dir / 'logistic_metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"✅ Метрики сохранены в {metrics_path}")
        
        return model, metrics


def main():
    args = parse_args()
    trainer = LogisticTrainer(config_path=args.config)
    trainer.run(feature_type=args.features)


if __name__ == "__main__":
    main()