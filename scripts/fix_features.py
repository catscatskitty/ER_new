#!/usr/bin/env python3
"""
Скрипт для проверки и исправления признаков
Путь: scripts/fix_features.py
"""

import numpy as np
from pathlib import Path

FEATURE_DIM = 38


def check_features():
    """Проверка файлов с признаками"""
    print("="*60)
    print("🔍 ПРОВЕРКА ФАЙЛОВ ПРИЗНАКОВ")
    print("="*60)
    
    processed_dir = Path('data/processed')
    
    for split_name in ['train', 'val', 'test']:
        features_file = processed_dir / f'features_{split_name}.npy'
        labels_file = processed_dir / f'labels_{split_name}.npy'
        
        if features_file.exists() and labels_file.exists():
            try:
                X = np.load(features_file)
                y = np.load(labels_file)
                
                print(f"\n{split_name.upper()}:")
                print(f"  Форма: {X.shape}")
                print(f"  Тип: {X.dtype}")
                print(f"  Метки: {np.bincount(y)}")
                
                if X.shape[1] != FEATURE_DIM:
                    print(f"  ❌ ОШИБКА: размерность {X.shape[1]} != {FEATURE_DIM}")
                    
                    # Создаем резервную копию
                    backup = features_file.with_suffix('.npy.bak')
                    features_file.rename(backup)
                    print(f"  📦 Создана резервная копия: {backup}")
                    
                    print(f"  💡 Запустите заново: python src/features/build_feature_pipeline.py --sequential")
                else:
                    print(f"  ✅ OK")
                    
            except Exception as e:
                print(f"  ❌ Ошибка при загрузке: {e}")
        else:
            print(f"\n{split_name.upper()}: ⚠️ Файлы не найдены")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    check_features()