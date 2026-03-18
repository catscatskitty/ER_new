#!/usr/bin/env python3
"""
Скрипт для обучения всех моделей с поддержкой --config
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
                       choices=['acoustic', 'linguistic', 'combined'],
                       help='Тип признаков для обучения')
    parser.add_argument('--model-type', type=str, default='all',
                       choices=['neural', 'traditional', 'all'],
                       help='Тип моделей для обучения')
    parser.add_argument('--force', action='store_true', help='Принудительное обучение')
    parser.add_argument('--skip-existing', action='store_true', help='Пропустить существующие')
    return parser.parse_args()


def check_model_exists(model_path):
    """Проверка существования модели"""
    return Path(model_path).exists()


def main():
    args = parse_args()
    logger = setup_logger('train_all')
    
    # Список моделей для обучения
    neural_models = [
        {'name': 'CNN', 'script': 'src/models/deep_learning/train_cnn_gpu.py', 
         'check': 'results/trained_models/torch_models/best_cnn.pth'},
        {'name': 'LSTM', 'script': 'src/models/deep_learning/train_lstm_gpu.py',
         'check': 'results/trained_models/torch_models/best_lstm.pth'},
        {'name': 'Hybrid', 'script': 'src/models/deep_learning/train_hybrid_gpu.py',
         'check': 'results/trained_models/torch_models/best_hybrid.pth'},
    ]
    
    traditional_models = [
        {'name': 'Logistic', 'script': 'src/models/traditional/train_logistic.py',
         'check': 'results/trained_models/logistic/model.pkl'},
        {'name': 'Random Forest', 'script': 'src/models/traditional/train_random_forest.py',
         'check': 'results/trained_models/random_forest/model.pkl'},
        {'name': 'XGBoost', 'script': 'src/models/traditional/train_xgboost.py',
         'check': 'results/trained_models/xgboost/model.pkl'},
        {'name': 'CatBoost', 'script': 'src/models/traditional/train_catboost.py',
         'check': 'results/trained_models/catboost/model.pkl'},
    ]
    
    # Выбор моделей
    models_to_train = []
    if args.model_type in ['neural', 'all']:
        models_to_train.extend(neural_models)
    if args.model_type in ['traditional', 'all']:
        models_to_train.extend(traditional_models)
    
    logger.info("=" * 60)
    logger.info(f"ОБУЧЕНИЕ МОДЕЛЕЙ")
    logger.info(f"Конфиг: {args.config}")
    logger.info(f"Признаки: {args.features}")
    logger.info(f"Тип моделей: {args.model_type}")
    logger.info(f"Всего моделей: {len(models_to_train)}")
    logger.info("=" * 60)
    
    results = []
    
    for i, model in enumerate(models_to_train, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i}/{len(models_to_train)}] Обучение: {model['name']}")
        logger.info(f"{'='*60}")
        
        # Проверка существования модели
        if args.skip_existing and check_model_exists(model['check']) and not args.force:
            logger.info(f"⏭️ Модель уже существует, пропускаем")
            results.append({'name': model['name'], 'status': 'skipped'})
            continue
        
        # Формирование команды
        cmd = [
            sys.executable, 
            model['script'], 
            '--config', args.config,
            '--features', args.features
        ]
        
        if args.force:
            cmd.append('--force')
        
        logger.info(f"Выполняем: {' '.join(cmd)}")
        
        # Запуск обучения
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        # Вывод результата
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        if result.returncode == 0:
            logger.info(f"✅ {model['name']} обучена за {elapsed:.2f} сек")
            results.append({'name': model['name'], 'status': 'success', 'time': elapsed})
        else:
            logger.error(f"❌ Ошибка при обучении {model['name']} (код: {result.returncode})")
            results.append({'name': model['name'], 'status': 'failed'})
            # При желании можно прервать выполнение
            # break
        
        # Пауза между моделями
        if i < len(models_to_train):
            logger.info("\n⏳ Пауза 3 секунды...")
            time.sleep(3)
    
    # Итог
    logger.info("\n" + "=" * 60)
    logger.info("ИТОГИ ОБУЧЕНИЯ")
    logger.info("=" * 60)
    
    success = sum(1 for r in results if r['status'] == 'success')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    failed = sum(1 for r in results if r['status'] == 'failed')
    
    logger.info(f"✅ Успешно: {success}")
    logger.info(f"⏭️ Пропущено: {skipped}")
    logger.info(f"❌ Ошибок: {failed}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()