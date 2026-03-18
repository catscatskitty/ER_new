#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

# ЖЕСТКО ЗАДАЕМ КОРЕНЬ ПРОЕКТА (работает всегда)
project_root = Path("C:/Users/Владелец/Desktop/ertelecom")
sys.path.insert(0, str(project_root))

print(f"✅ КОРЕНЬ ПРОЕКТА: {project_root}")
print(f"✅ Python path: {sys.path[0]}")

# Теперь импортируем
import numpy as np
import logging
import yaml
import argparse
from tqdm import tqdm
import librosa

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_audio(file_path, target_sr=8000):
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return audio, sr
    except Exception as e:
        logger.error(f"Ошибка загрузки {file_path}: {e}")
        return None, None

def extract_features(audio, sr, config):
    features = []
    
    n_mfcc = config.get('acoustic', {}).get('mfcc', {}).get('n_mfcc', 20)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))
    
    cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features.append(np.mean(cent))
    features.append(np.std(cent))
    
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    features.append(np.mean(bandwidth))
    features.append(np.std(bandwidth))
    
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))
    
    zcr = librosa.feature.zero_crossing_rate(audio)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))
    
    rms = librosa.feature.rms(y=audio)
    features.append(np.mean(rms))
    features.append(np.std(rms))
    
    return np.array(features)

def build_acoustic_features(config_path: str = "configs/feature_config.yaml"):
    """
    Сбор всех акустических признаков
    """
    logger.info("="*60)
    logger.info("ИЗВЛЕЧЕНИЕ АКУСТИЧЕСКИХ ПРИЗНАКОВ")
    logger.info("="*60)
    
    # Загружаем конфиг - путь относительно корня проекта
    config_full_path = project_root / config_path
    logger.info(f"Загрузка конфига: {config_full_path}")
    
    if not config_full_path.exists():
        logger.error(f"❌ Конфиг не найден: {config_full_path}")
        logger.info(f"\nСоздайте конфиг по пути: {config_full_path}")
        logger.info("Или проверьте наличие папки configs в корне проекта")
        return
    
    with open(config_full_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Проверяем директории
    processed_dir = project_root / 'data' / 'processed'
    splits_dir = project_root / 'data' / 'splits'
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    if not splits_dir.exists():
        logger.error(f"❌ Директория splits не найдена: {splits_dir}")
        logger.info("Сначала запустите:")
        logger.info("  python src/data_preparation/04_create_splits.py")
        return
    
    # Загружаем списки файлов
    splits = {}
    for split_name in ['train', 'val', 'test']:
        split_file = splits_dir / f'{split_name}_files.txt'
        if not split_file.exists():
            logger.error(f"❌ Файл не найден: {split_file}")
            return
        
        with open(split_file, 'r', encoding='utf-8') as f:
            files = [line.strip() for line in f.readlines() if line.strip()]
            splits[split_name] = files
        
        logger.info(f"{split_name}: {len(files)} файлов")
    
    # Извлекаем признаки
    for split_name, file_list in splits.items():
        logger.info(f"\nИзвлечение признаков для {split_name}...")
        
        features = []
        labels = []
        
        for file_path in tqdm(file_list, desc=f"Обработка {split_name}"):
            audio_path = Path(file_path)
            if not audio_path.exists():
                logger.warning(f"Файл не найден: {audio_path}")
                continue
            
            label = 1 if 'robot' in str(file_path).lower() else 0
            
            audio, sr = load_audio(str(audio_path), target_sr=8000)
            if audio is None:
                continue
            
            try:
                feature_vector = extract_features(audio, sr, config)
                features.append(feature_vector)
                labels.append(label)
            except Exception as e:
                logger.error(f"Ошибка: {e}")
                continue
        
        if features:
            features_array = np.array(features)
            labels_array = np.array(labels)
            
            np.save(processed_dir / f'features_{split_name}.npy', features_array)
            np.save(processed_dir / f'labels_{split_name}.npy', labels_array)
            
            logger.info(f"{split_name}: {features_array.shape[0]} samples, {features_array.shape[1]} features")
            logger.info(f"  Human: {(labels_array == 0).sum()}, Robot: {(labels_array == 1).sum()}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ ЗАВЕРШЕНО")
    logger.info("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/feature_config.yaml')
    args = parser.parse_args()
    
    build_acoustic_features(args.config)