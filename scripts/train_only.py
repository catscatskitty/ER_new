#!/usr/bin/env python
"""
Только обучение моделей (без подготовки данных)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def train_only(config_path: str = "configs/training_config.yaml"):
    """
    Запуск только обучения моделей
    """
    print("=" * 60)
    print("ЗАПУСК ОБУЧЕНИЯ МОДЕЛЕЙ")
    print("=" * 60)
    
    # Проверка наличия данных
    processed_dir = Path('data/processed')
    required_files = ['features_train.npy', 'labels_train.npy']
    
    missing_files = []
    for f in required_files:
        if not (processed_dir / f).exists():
            missing_files.append(f)
    
    if missing_files:
        print("❌ Ошибка: Отсутствуют файлы с признаками:")
        for f in missing_files:
            print(f"  - data/processed/{f}")
        print("\nСначала запустите подготовку данных:")
        print("  python scripts/run_full_pipeline.py")
        return False
    
    # Запуск обучения
    cmd = [sys.executable, 'src/training/train_all_models.py', '--config', config_path]
    
    try:
        subprocess.run(cmd)
        return True
    except Exception as e:
        print(f"Ошибка при обучении: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/training_config.yaml')
    args = parser.parse_args()
    
    train_only(args.config)