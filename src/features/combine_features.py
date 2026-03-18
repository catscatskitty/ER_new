#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ОБЪЕДИНЕНИЕ АКУСТИЧЕСКИХ И ФОНЕТИЧЕСКИХ ПРИЗНАКОВ
Создает единые фичи для всех моделей
"""

import os
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def combine_features():
    """
    Объединение acoustic + phonetic = combined
    """
    print("="*60)
    print("ОБЪЕДИНЕНИЕ ПРИЗНАКОВ")
    print("="*60)
    
    processed_dir = project_root / 'data' / 'processed'
    
    acoustic_shape = None
    phonetic_shape = None
    
    for split in ['train', 'val', 'test']:
        acoustic_file = processed_dir / f'features_{split}.npy'
        phonetic_file = processed_dir / f'phonetic_{split}.npy'
        
        if not acoustic_file.exists():
            print(f"❌ Нет акустических признаков для {split}")
            continue
            
        if not phonetic_file.exists():
            print(f"❌ Нет фонетических признаков для {split}")
            continue
        
        acoustic = np.load(acoustic_file)
        phonetic = np.load(phonetic_file)
        
        if acoustic_shape is None:
            acoustic_shape = acoustic.shape[1]
            phonetic_shape = phonetic.shape[1]
        
        # Проверка соответствия
        if len(acoustic) != len(phonetic):
            print(f"⚠️  Несоответствие размеров: acoustic {len(acoustic)}, phonetic {len(phonetic)}")
            min_len = min(len(acoustic), len(phonetic))
            acoustic = acoustic[:min_len]
            phonetic = phonetic[:min_len]
        
        # ОБЪЕДИНЕНИЕ
        combined = np.concatenate([acoustic, phonetic], axis=1)
        
        print(f"\n{split}:")
        print(f"  Акустика: {acoustic.shape}")
        print(f"  Фонетика: {phonetic.shape}")
        print(f"  Объединенные: {combined.shape}")
        
        # Сохраняем
        combined_file = processed_dir / f'combined_{split}.npy'
        np.save(combined_file, combined)
        
        # Копируем метки
        labels_file = processed_dir / f'labels_{split}.npy'
        if labels_file.exists():
            labels = np.load(labels_file)
            if len(labels) > len(combined):
                labels = labels[:len(combined)]
            np.save(processed_dir / f'combined_labels_{split}.npy', labels)
    
    print(f"\n✅ Всего признаков: {acoustic_shape + phonetic_shape}")
    print(f"   Акустических: {acoustic_shape}")
    print(f"   Фонетических: {phonetic_shape}")
    
    # Создаем описание
    desc_file = processed_dir / 'feature_description.txt'
    with open(desc_file, 'w', encoding='utf-8') as f:
        f.write(f"Total features: {acoustic_shape + phonetic_shape}\n")
        f.write(f"Acoustic (0-{acoustic_shape-1}): MFCC, spectral, prosodic\n")
        f.write(f"Phonetic ({acoustic_shape}-{acoustic_shape+phonetic_shape-1}): F0, jitter, shimmer, formants, etc.\n")
    
    print(f"✅ Описание сохранено: {desc_file}")

if __name__ == "__main__":
    combine_features()