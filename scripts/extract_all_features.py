#!/usr/bin/env python
"""
Доизвлечение признаков для val и test (train уже есть)
Использует multiprocessing для ускорения
"""

import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.features.acoustic.extract_acoustic_features import AcousticFeatureExtractor
from src.features.phonetic.extract_phonetic_features import PhoneticFeatureExtractor
from src.features.spectrogram.extract_spectrogram import SpectrogramExtractor
from src.features.mfcc.extract_mfcc_sequence import MFCCSequenceExtractor


def load_split_files(split_dir, split_name):
    """Загрузка списка файлов для разбиения (относительные пути)"""
    split_file = split_dir / f"{split_name}_files.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Missing {split_file}")
    
    with open(split_file, 'r') as f:
        return [line.strip() for line in f.readlines()]


def check_existing_features(output_dir, split_name):
    """Проверка, какие файлы уже существуют для данного split"""
    existing = {}
    for subdir in ['acoustic', 'phonetic', 'combined', 'spectrograms', 'mfcc_sequences']:
        feat_path = output_dir / subdir / f'features_{split_name}.npy'
        label_path = output_dir / subdir / f'labels_{split_name}.npy'
        if feat_path.exists() and label_path.exists():
            existing[subdir] = True
            print(f"  ✅ {subdir} уже существует")
        else:
            existing[subdir] = False
    return existing


def process_split(split_name, files, audio_dir, output_dir, force=False):
    """Обработка одного разбиения с проверкой существующих файлов"""
    print(f"\n{'='*50}")
    print(f"📊 Processing {split_name} split ({len(files)} files)")
    print(f"{'='*50}")
    
    # Проверяем, что уже извлечено
    existing = check_existing_features(output_dir, split_name)
    
    # Если всё уже есть и не force, пропускаем
    if not force and all(existing.values()):
        print(f"✅ {split_name} полностью извлечён, пропускаем")
        return True
    
    audio_paths = [audio_dir / f for f in files]
    
    # Извлекаем только недостающее
    if not existing.get('acoustic', False) or force:
        print("  🎵 Extracting acoustic features...")
        acoustic_ext = AcousticFeatureExtractor()
        acoustic = acoustic_ext.extract_batch_parallel(
            audio_paths, 
            output_dir / 'acoustic' / f'features_{split_name}.npy'
        )
    else:
        # Загружаем существующие для объединения
        acoustic = np.load(output_dir / 'acoustic' / f'features_{split_name}.npy')
        print(f"  🎵 Using existing acoustic: {acoustic.shape}")
    
    if not existing.get('phonetic', False) or force:
        print("  📝 Extracting phonetic features...")
        phonetic_ext = PhoneticFeatureExtractor()
        phonetic = phonetic_ext.extract_batch_parallel(
            audio_paths,
            output_dir / 'phonetic' / f'phonetic_{split_name}.npy'
        )
    else:
        phonetic = np.load(output_dir / 'phonetic' / f'phonetic_{split_name}.npy')
        print(f"  📝 Using existing phonetic: {phonetic.shape}")
    
    # Комбинированные всегда пересоздаём (если есть акустика и фонетика)
    if (existing.get('acoustic', False) or not force) and (existing.get('phonetic', False) or not force):
        print("  🔗 Creating combined features...")
        combined = np.concatenate([acoustic, phonetic], axis=1)
        np.save(output_dir / 'combined' / f'combined_{split_name}.npy', combined)
        print(f"  ✅ Combined: {combined.shape}")
    
    # Спектрограммы
    if not existing.get('spectrograms', False) or force:
        print("  🌊 Extracting spectrograms...")
        spec_ext = SpectrogramExtractor()
        spec_ext.extract_batch_parallel(
            audio_paths,
            output_dir / 'spectrograms' / f'spectrograms_{split_name}.npy'
        )
    
    # MFCC-последовательности
    if not existing.get('mfcc_sequences', False) or force:
        print("  📈 Extracting MFCC sequences...")
        mfcc_ext = MFCCSequenceExtractor()
        mfcc_ext.extract_batch_parallel(
            audio_paths,
            output_dir / 'mfcc_sequences' / f'mfcc_{split_name}.npy'
        )
    
    # Метки (всегда пересоздаём, это быстро)
    labels = np.array([0 if 'human' in f else 1 for f in files])
    for subdir in ['acoustic', 'phonetic', 'combined', 'spectrograms', 'mfcc_sequences']:
        np.save(output_dir / subdir / f'labels_{split_name}.npy', labels)
    
    print(f"✅ {split_name} completed!")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Доизвлечение признаков')
    parser.add_argument('--splits', nargs='+', default=['val', 'test'],
                       choices=['train', 'val', 'test'],
                       help='Какие splits обработать (по умолчанию val test)')
    parser.add_argument('--force', action='store_true',
                       help='Принудительно пересоздать все файлы')
    args = parser.parse_args()
    
    # Пути
    audio_dir = Path('data/processed/augmented_8khz')
    split_dir = Path('data/splits')
    output_dir = Path('data/processed')
    
    # Проверка наличия split файлов
    for split in args.splits:
        split_file = split_dir / f"{split}_files.txt"
        if not split_file.exists():
            print(f"❌ Missing {split_file}")
            sys.exit(1)
    
    # Загрузка разбиений
    splits = {}
    for split in args.splits:
        splits[split] = load_split_files(split_dir, split)
    
    print(f"\n📂 Loading splits:")
    for name, files in splits.items():
        print(f"  {name}: {len(files)} files")
    
    # Создаём директории, если их нет
    for subdir in ['acoustic', 'phonetic', 'combined', 'spectrograms', 'mfcc_sequences']:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Обрабатываем каждый split
    for split_name, files in splits.items():
        try:
            process_split(split_name, files, audio_dir, output_dir, args.force)
        except Exception as e:
            print(f"❌ Ошибка при обработке {split_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n🎉 Все признаки успешно извлечены!")


if __name__ == "__main__":
    main()