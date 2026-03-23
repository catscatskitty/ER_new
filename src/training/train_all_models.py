#!/usr/bin/env python3
"""
Скрипт для обучения всех моделей (нейросетевых и традиционных) с поддержкой
выбора типа признаков (acoustic, phonetic, combined) и типа моделей.
Запускает соответствующие скрипты обучения.
"""

import argparse
import sys
import subprocess
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Обучение всех моделей')
    parser.add_argument('--config', type=str, default='configs', help='Путь к конфигам')
    parser.add_argument('--features', type=str, default='acoustic',
                       choices=['acoustic', 'phonetic', 'combined'],
                       help='Тип признаков для обучения')
    parser.add_argument('--model-type', type=str, default='all',
                       choices=['neural', 'traditional', 'all'],
                       help='Тип моделей для обучения')
    parser.add_argument('--force', action='store_true', help='Принудительное переобучение')
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger('train_all')

    # Список моделей и соответствующих скриптов
    # Нейросетевые модели (акустические и комбинированные)
    neural_models = [
        {'name': 'CNN', 'script': 'src/models/deep_learning/train_spectrogram_cnn.py',
         'check': 'results/trained_models/torch_models/best_cnn.pth'},
        {'name': 'LSTM', 'script': 'src/models/deep_learning/train_mfcc_lstm.py',
         'check': 'results/trained_models/torch_models/best_lstm.pth'},
        {'name': 'Hybrid', 'script': 'src/models/deep_learning/train_multimodal.py',
         'check': 'results/trained_models/torch_models/best_hybrid.pth'},
        {'name': 'MLP', 'script': 'src/models/deep_learning/train_mlp_gpu.py',
         'check': 'results/trained_models/torch_models/best_mlp.pth'},
    ]

    # Традиционные модели
    traditional_models = [
        {'name': 'Logistic', 'script': 'src/models/traditional/train_logistic.py',
         'check': 'results/trained_models/logistic/model.pkl'},
        {'name': 'RandomForest', 'script': 'src/models/traditional/train_random_forest.py',
         'check': 'results/trained_models/random_forest/model.pkl'},
        {'name': 'XGBoost', 'script': 'src/models/traditional/train_xgboost.py',
         'check': 'results/trained_models/xgboost/model.pkl'},
        {'name': 'CatBoost', 'script': 'src/models/traditional/train_catboost.py',
         'check': 'results/trained_models/catboost/model.pkl'},
    ]

    # Определяем список моделей для обучения
    if args.model_type == 'neural':
        models_to_train = neural_models
    elif args.model_type == 'traditional':
        models_to_train = traditional_models
    else:
        models_to_train = neural_models + traditional_models

    logger.info("=" * 60)
    logger.info("ОБУЧЕНИЕ ВСЕХ МОДЕЛЕЙ")
    logger.info(f"Конфиг: {args.config}")
    logger.info(f"Тип признаков: {args.features}")
    logger.info(f"Тип моделей: {args.model_type}")
    logger.info(f"Всего моделей: {len(models_to_train)}")
    logger.info("=" * 60)

    for i, model in enumerate(models_to_train, 1):
        logger.info(f"\n--- Прогресс: {i}/{len(models_to_train)} ---")
        logger.info(f"Обучение модели: {model['name']}")

        # Формируем команду
        cmd = [
            sys.executable,
            model['script'],
            '--config', args.config,
            '--features', args.features
        ]
        if args.force:
            cmd.append('--force')

        logger.info(f"Выполняем: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)

            if result.returncode != 0:
                logger.error(f"❌ Ошибка при обучении {model['name']} (код: {result.returncode})")
                # Можно остановить выполнение при ошибке
                # break
            else:
                logger.info(f"✅ {model['name']} обучена")

        except Exception as e:
            logger.error(f"❌ Исключение при обучении {model['name']}: {e}")

        # Пауза между моделями
        if i < len(models_to_train):
            time.sleep(2)

    logger.info("\n" + "=" * 60)
    logger.info("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()