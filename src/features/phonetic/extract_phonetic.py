#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPU-УСКОРЕННОЕ ИЗВЛЕЧЕНИЕ ФОНЕТИЧЕСКИХ ПРИЗНАКОВ НА PYTORCH
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Жестко задаем корень проекта
project_root = Path("C:/Users/Владелец/Desktop/ertelecom")
sys.path.insert(0, str(project_root))

print(f"✅ Корень проекта: {project_root}")

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import librosa
from scipy import stats
from scipy.signal import find_peaks
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import time
import argparse
import logging

# Проверка GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Используется устройство: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPUPhoneticExtractor:
    """
    Извлечение фонетических признаков на GPU через PyTorch
    """
    
    def __init__(self, sr=8000, batch_size=32):
        self.sr = sr
        self.batch_size = batch_size
        self.device = device
        self.feature_names = [
            'f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_range', 'f0_cv',
            'voiced_ratio', 'jitter', 'shimmer', 'hnr',
            'centroid_mean', 'centroid_std',
            'bandwidth_mean', 'bandwidth_std',
            'rolloff_mean', 'rolloff_std',
            'mfcc1_mean', 'mfcc1_std', 'mfcc2_mean', 'mfcc2_std',
            'mfcc3_mean', 'mfcc3_std', 'mfcc4_mean', 'mfcc4_std',
            'mfcc5_mean', 'mfcc5_std',
            'speech_ratio', 'speech_segments', 'pauses',
            'rms_mean', 'rms_std', 'rms_cv',
            'band1_energy', 'band2_energy', 'band3_energy', 'band4_energy',
            'hf_noise_ratio', 'amp_jumps', 'spectral_flatness'
        ]
        
    def spectrogram_gpu(self, audio_batch):
        """
        Вычисление спектрограммы на GPU
        audio_batch: list of numpy arrays
        """
        batch_size = len(audio_batch)
        max_len = max(len(a) for a in audio_batch)
        
        # Подготавливаем тензор [batch, max_len]
        audio_tensor = torch.zeros(batch_size, max_len, device=self.device)
        for i, a in enumerate(audio_batch):
            audio_tensor[i, :len(a)] = torch.from_numpy(a).to(self.device)
        
        # Параметры STFT
        n_fft = 512
        hop_length = 256
        
        # Вычисляем STFT для всего батча
        # torch.stft возвращает [batch, freq, time, 2]
        stft_complex = torch.stft(
            audio_tensor,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft).to(self.device),
            return_complex=True
        )
        
        # Амплитудная спектрограмма
        spec = torch.abs(stft_complex)  # [batch, freq, time]
        
        return spec
    
    def compute_mfcc_gpu(self, spec, n_mfcc=5):
        """
        Вычисление MFCC на GPU
        """
        batch_size, n_freq, n_time = spec.shape
        
        # Mel фильтры (упрощенно)
        mel_basis = torch.linspace(0, 1, n_freq, device=self.device).view(1, -1, 1)
        mel_spec = torch.sum(spec * mel_basis, dim=1)  # [batch, time]
        
        # Log и DCT (аппроксимация)
        log_mel = torch.log(mel_spec + 1e-10)
        
        # Простая аппроксимация DCT
        mfcc = []
        for i in range(n_mfcc):
            basis = torch.cos(torch.pi * i * torch.arange(n_time, device=self.device) / n_time)
            coeff = torch.mean(log_mel * basis, dim=1)
            mfcc.append(coeff)
        
        return torch.stack(mfcc, dim=1)  # [batch, n_mfcc]
    
    def extract_batch(self, batch_files):
        """
        Обработка батча файлов на GPU
        """
        # Загружаем аудио
        audio_batch = []
        valid_paths = []
        labels = []
        
        for file_path, label in batch_files:
            try:
                y, _ = librosa.load(file_path, sr=self.sr, mono=True, duration=5.0)
                if len(y) > 0.1 * self.sr:
                    # Нормализуем длину
                    target_len = 5 * self.sr
                    if len(y) > target_len:
                        y = y[:target_len]
                    elif len(y) < target_len:
                        y = np.pad(y, (0, target_len - len(y)))
                    
                    audio_batch.append(y)
                    valid_paths.append(file_path)
                    labels.append(label)
            except Exception as e:
                logger.error(f"Ошибка загрузки {file_path}: {e}")
        
        if not audio_batch:
            return []
        
        # Вычисляем спектрограмму на GPU
        spec = self.spectrogram_gpu(audio_batch)  # [batch, freq, time]
        
        # Вычисляем признаки на GPU
        batch_size = len(audio_batch)
        
        # RMS энергия
        rms = torch.sqrt(torch.mean(spec**2, dim=1))  # [batch, time]
        rms_mean = torch.mean(rms, dim=1).cpu().numpy()
        rms_std = torch.std(rms, dim=1).cpu().numpy()
        
        # Спектральный центроид
        freqs = torch.linspace(0, self.sr/2, spec.shape[1], device=self.device).view(1, -1, 1)
        centroid = torch.sum(spec * freqs, dim=1) / (torch.sum(spec, dim=1) + 1e-10)
        centroid_mean = torch.mean(centroid, dim=1).cpu().numpy()
        centroid_std = torch.std(centroid, dim=1).cpu().numpy()
        
        # Спектральная ширина
        bandwidth = torch.sqrt(torch.sum(spec * (freqs - centroid.unsqueeze(1))**2, dim=1) / 
                              (torch.sum(spec, dim=1) + 1e-10))
        bandwidth_mean = torch.mean(bandwidth, dim=1).cpu().numpy()
        bandwidth_std = torch.std(bandwidth, dim=1).cpu().numpy()
        
        # MFCC
        mfcc = self.compute_mfcc_gpu(spec, n_mfcc=5)  # [batch, 5]
        mfcc_mean = mfcc.cpu().numpy()
        
        # Энергия по частотным полосам
        n_freq = spec.shape[1]
        band_size = n_freq // 4
        band_energy = []
        for i in range(4):
            start = i * band_size
            end = (i + 1) * band_size if i < 3 else n_freq
            energy = torch.mean(spec[:, start:end, :], dim=(1, 2)).cpu().numpy()
            band_energy.append(energy)
        
        # VAD (простейший)
        speech_ratio = (rms > torch.mean(rms, dim=1, keepdim=True) * 0.3).float().mean(dim=1).cpu().numpy()
        
        # Формируем результаты
        results = []
        for i in range(batch_size):
            features = {
                'filepath': str(valid_paths[i]),
                'filename': Path(valid_paths[i]).name,
                'label': labels[i],
                
                'rms_mean': float(rms_mean[i]),
                'rms_std': float(rms_std[i]),
                'rms_cv': float(rms_std[i] / (rms_mean[i] + 1e-10)),
                
                'centroid_mean': float(centroid_mean[i]),
                'centroid_std': float(centroid_std[i]),
                
                'bandwidth_mean': float(bandwidth_mean[i]),
                'bandwidth_std': float(bandwidth_std[i]),
                
                'speech_ratio': float(speech_ratio[i]),
                'speech_segments': 0,  # TODO
                'pauses': 0,  # TODO
                
                'band1_energy': float(band_energy[0][i]),
                'band2_energy': float(band_energy[1][i]),
                'band3_energy': float(band_energy[2][i]),
                'band4_energy': float(band_energy[3][i]),
                'hf_noise_ratio': float(band_energy[3][i] / (band_energy[0][i] + 1e-10)),
            }
            
            # MFCC
            for j in range(5):
                features[f'mfcc{j+1}_mean'] = float(mfcc_mean[i, j])
                features[f'mfcc{j+1}_std'] = 0.0  # TODO
            
            # Заглушки для остальных признаков
            features.update({
                'f0_mean': 0.0, 'f0_std': 0.0, 'f0_min': 0.0, 'f0_max': 0.0,
                'f0_range': 0.0, 'f0_cv': 0.0, 'voiced_ratio': 0.0,
                'jitter': 0.0, 'shimmer': 0.0, 'hnr': 0.0,
                'amp_jumps': 0.0, 'spectral_flatness': 0.0,
            })
            
            results.append(features)
        
        return results


def process_batch_wrapper(batch_files, extractor):
    """Обертка для обработки батча"""
    return extractor.extract_batch(batch_files)


def extract_phonetic_features_gpu(split='train', batch_size=32, num_workers=4):
    """
    GPU-ускоренное извлечение признаков
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"GPU ИЗВЛЕЧЕНИЕ: {split.upper()}")
    logger.info(f"{'='*60}")
    
    # Путь к сплитам
    splits_dir = project_root / 'data' / 'splits'
    split_file = splits_dir / f'{split}_files.txt'
    
    if not split_file.exists():
        logger.error(f"❌ Файл не найден: {split_file}")
        return None
    
    with open(split_file, 'r', encoding='utf-8') as f:
        file_paths = [line.strip() for line in f.readlines() if line.strip()]
    
    logger.info(f"Найдено {len(file_paths)} файлов в сплите")
    
    # Фильтруем существующие файлы с метками
    valid_files = []
    for file_path in file_paths:
        path = Path(file_path)
        if path.exists():
            label = 1 if 'robot' in str(path).lower() else 0
            valid_files.append((path, label))
        else:
            logger.warning(f"Файл не существует: {path}")
    
    logger.info(f"Реально существует: {len(valid_files)} из {len(file_paths)} файлов")
    
    if not valid_files:
        logger.error("❌ Нет доступных файлов!")
        return None
    
    # Разбиваем на батчи
    batches = [valid_files[i:i+batch_size] for i in range(0, len(valid_files), batch_size)]
    logger.info(f"Создано {len(batches)} батчей по {batch_size} файлов")
    
    # Создаем экстрактор
    extractor = GPUPhoneticExtractor(sr=8000, batch_size=batch_size)
    
    # Обрабатываем батчи параллельно
    all_results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_batch_wrapper, batch, extractor) for batch in batches]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Обработка батчей"):
            try:
                batch_results = future.result(timeout=60)
                all_results.extend(batch_results)
            except Exception as e:
                logger.error(f"Ошибка батча: {e}")
    
    elapsed = time.time() - start_time
    
    logger.info(f"✅ Успешно обработано: {len(all_results)}/{len(valid_files)}")
    logger.info(f"⏱️  Время: {elapsed:.1f} сек, {len(all_results)/elapsed:.1f} файлов/сек")
    
    if not all_results:
        logger.error("❌ Нет результатов!")
        return None
    
    # Создаем DataFrame
    df = pd.DataFrame(all_results)
    
    # Сохраняем CSV
    csv_dir = project_root / 'data' / 'phonetic_features'
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f'phonetic_{split}.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"✅ CSV сохранен: {csv_path}")
    
    # Сохраняем numpy
    feature_cols = [c for c in df.columns if c not in ['filepath', 'filename', 'label']]
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['label'].values.astype(np.int32)
    
    processed_dir = project_root / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(processed_dir / f'phonetic_{split}.npy', X)
    np.save(processed_dir / f'phonetic_labels_{split}.npy', y)
    
    logger.info(f"\n📊 Статистика {split}:")
    logger.info(f"  X shape: {X.shape}")
    logger.info(f"  y shape: {y.shape}")
    logger.info(f"  Human: {np.sum(y == 0)}")
    logger.info(f"  Robot: {np.sum(y == 1)}")
    
    return X, y


def main():
    parser = argparse.ArgumentParser(description='GPU-ускоренное извлечение фонетических признаков')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                       help='Сплиты для обработки')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Размер батча для GPU')
    parser.add_argument('--workers', type=int, default=4,
                       help='Количество потоков для загрузки')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("GPU ИЗВЛЕЧЕНИЕ ФОНЕТИЧЕСКИХ ПРИЗНАКОВ")
    logger.info("="*60)
    logger.info(f"Корень проекта: {project_root}")
    logger.info(f"Устройство: {device}")
    logger.info(f"Батч-размер: {args.batch_size}")
    logger.info(f"Потоков: {args.workers}")
    logger.info("="*60)
    
    # Проверяем наличие сплитов
    splits_dir = project_root / 'data' / 'splits'
    if not splits_dir.exists():
        logger.error(f"❌ Директория сплитов не существует: {splits_dir}")
        logger.info("Создайте сплиты командой:")
        logger.info(f"  python src/data_preparation/04_create_splits.py")
        return
    
    total_start = time.time()
    
    for split in args.splits:
        extract_phonetic_features_gpu(split, args.batch_size, args.workers)
    
    total_elapsed = time.time() - total_start
    logger.info(f"\n⏱️  ОБЩЕЕ ВРЕМЯ: {total_elapsed:.1f} сек")
    
    logger.info("\n" + "="*60)
    logger.info("✅ ВСЕ ПРИЗНАКИ ИЗВЛЕЧЕНЫ")
    logger.info("="*60)


if __name__ == "__main__":
    main()