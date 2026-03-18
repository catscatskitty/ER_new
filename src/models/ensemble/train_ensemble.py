#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import logging
import json
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent  # ensemble -> models -> src -> ertelecom
sys.path.insert(0, str(project_root))

from src.models.ensemble.voting_classifier import VotingEnsemble
from src.models.ensemble.stacking_classifier import StackingEnsemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_base_models(feature_type):
    """Загрузка всех обученных моделей"""
    models = {}
    models_dir = project_root / 'results' / 'trained_models'
    
    model_paths = {
        'logistic': models_dir / 'logistic' / f'logistic_{feature_type}.pkl',
        'random_forest': models_dir / 'random_forest' / f'random_forest_{feature_type}.pkl',
        'xgboost': models_dir / 'xgboost' / f'xgboost_{feature_type}.pkl',
        'catboost': models_dir / 'catboost' / f'catboost_{feature_type}.cbm',
    }
    
    for name, path in model_paths.items():
        if path.exists():
            try:
                if name == 'catboost':
                    from catboost import CatBoostClassifier
                    models[name] = CatBoostClassifier()
                    models[name].load_model(str(path))
                else:
                    models[name] = joblib.load(path)
                logger.info(f"✅ Загружена модель: {name}")
            except Exception as e:
                logger.warning(f"❌ Не удалось загрузить {name}: {e}")
    
    return list(models.values())

def load_features(feature_type, split):
    processed_dir = project_root / 'data' / 'processed'
    
    if feature_type == 'acoustic':
        file_path = processed_dir / f'features_{split}.npy'
    elif feature_type == 'phonetic':
        file_path = processed_dir / f'phonetic_{split}.npy'
    else:
        file_path = processed_dir / f'combined_{split}.npy'
    
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    X = np.load(file_path)
    y = np.load(processed_dir / f'labels_{split}.npy')
    
    if len(X) > len(y):
        X = X[:len(y)]
    elif len(y) > len(X):
        y = y[:len(X)]
    
    return X, y

def train_ensemble(feature_type='combined'):
    logger.info("="*60)
    logger.info(f"АНСАМБЛИ - ПРИЗНАКИ: {feature_type}")
    logger.info("="*60)
    
    try:
        X_train, y_train = load_features(feature_type, 'train')
        X_val, y_val = load_features(feature_type, 'val')
        X_test, y_test = load_features(feature_type, 'test')
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        return
    
    # Нормализация
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1
    
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    
    # Загрузка базовых моделей
    base_models = load_base_models(feature_type)
    
    if len(base_models) < 2:
        logger.error("❌ Недостаточно моделей для ансамбля")
        return
    
    logger.info(f"Загружено {len(base_models)} моделей")
    
    # Voting Ensemble
    logger.info("\n--- Voting Ensemble ---")
    voting = VotingEnsemble(models=base_models, voting='soft')
    
    test_pred = voting.predict(X_test)
    test_proba = voting.predict_proba(X_test)
    
    voting_metrics = {
        'model': f'voting_{feature_type}',
        'feature_type': feature_type,
        'test_accuracy': float(accuracy_score(y_test, test_pred)),
        'test_f1': float(f1_score(y_test, test_pred, average='weighted')),
        'test_auc': float(roc_auc_score(y_test, test_proba[:, 1])),
        'confusion_matrix': confusion_matrix(y_test, test_pred).tolist()
    }
    
    logger.info(f"Voting Accuracy: {voting_metrics['test_accuracy']:.4f}")
    
    # Сохранение
    ensemble_dir = project_root / 'results' / 'trained_models' / 'ensemble'
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(voting, ensemble_dir / f'voting_{feature_type}.pkl')
    
    # Сохранение метрик
    metrics_dir = project_root / 'results' / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_dir / f'ensemble_{feature_type}_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(voting_metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Ансамбль сохранен")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, default='combined',
                       choices=['acoustic', 'phonetic', 'combined'])
    args = parser.parse_args()
    
    train_ensemble(args.features)