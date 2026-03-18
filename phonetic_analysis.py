#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
БЫСТРЫЙ ФОНЕТИЧЕСКИЙ АНАЛИЗ ДАТАСЕТА (CPU OPTIMIZED)
Только augmented_final + многопоточность
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Корень проекта
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import time
import numpy as np
import pandas as pd
import librosa
from scipy import stats
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import queue
import threading

# Настройка стиля
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Количество ядер CPU
CPU_COUNT = mp.cpu_count()
print(f"✅ Доступно ядер CPU: {CPU_COUNT}")

class FastFeatureExtractor:
    """
    Оптимизированное извлечение признаков (CPU)
    """
    
    def __init__(self, sr=8000):
        self.sr = sr
        
    def extract_features(self, file_path):
        """Извлечение признаков из одного файла"""
        try:
            # Загружаем аудио
            y, _ = librosa.load(file_path, sr=self.sr, mono=True, duration=5.0)
            
            if len(y) < 0.1 * self.sr:
                return None
            
            features = {}
            
            # 1. Базовые характеристики (очень быстро)
            features['duration'] = len(y) / self.sr
            features['rms'] = float(np.sqrt(np.mean(y**2)))
            features['peak'] = float(np.max(np.abs(y)))
            features['zero_crossings'] = float(np.sum(np.abs(np.diff(np.signbit(y)))) / len(y))
            
            # 2. Спектральные характеристики (быстро)
            spec = np.abs(librosa.stft(y, n_fft=512, hop_length=256))
            
            # Спектральный центроид
            cent = librosa.feature.spectral_centroid(S=spec, sr=self.sr)[0]
            features['centroid_mean'] = float(np.mean(cent))
            features['centroid_std'] = float(np.std(cent))
            
            # Спектральная ширина
            bandwidth = librosa.feature.spectral_bandwidth(S=spec, sr=self.sr)[0]
            features['bandwidth_mean'] = float(np.mean(bandwidth))
            
            # Спектральный спад
            rolloff = librosa.feature.spectral_rolloff(S=spec, sr=self.sr)[0]
            features['rolloff_mean'] = float(np.mean(rolloff))
            
            # 3. MFCC (только 5 коэффициентов для скорости)
            mfccs = librosa.feature.mfcc(S=librosa.power_to_db(spec), n_mfcc=5)
            for i, mfcc in enumerate(mfccs):
                features[f'mfcc_{i+1}_mean'] = float(np.mean(mfcc))
                features[f'mfcc_{i+1}_std'] = float(np.std(mfcc))
            
            # 4. Энергия по частотным полосам
            freqs = librosa.fft_frequencies(sr=self.sr, n_fft=512)
            spec_mean = np.mean(spec, axis=1)
            
            bands = [(50, 500), (500, 1500), (1500, 2500), (2500, 4000)]
            for i, (low, high) in enumerate(bands):
                mask = (freqs >= low) & (freqs < high)
                if np.any(mask):
                    features[f'band_{i+1}_energy'] = float(np.mean(spec_mean[mask]))
                else:
                    features[f'band_{i+1}_energy'] = 0.0
            
            # 5. F0 (упрощенно через автокорреляцию)
            autocorr = np.correlate(y, y, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Поиск пиков
            peaks = find_peaks(autocorr[:len(autocorr)//4], height=np.max(autocorr)*0.3)[0]
            if len(peaks) > 0:
                period = peaks[0]
                features['f0'] = float(self.sr / period) if period > 0 else 0
            else:
                features['f0'] = 0.0
            
            # 6. Jitter (микровариации периода)
            if len(peaks) > 5:
                periods = peaks[:10]  # Берем первые 10 пиков
                period_values = self.sr / periods
                features['jitter'] = float(np.std(period_values) / np.mean(period_values))
            else:
                features['jitter'] = 0.0
            
            return features
            
        except Exception as e:
            print(f"Ошибка: {file_path.name} - {e}")
            return None


def scan_augmented_final():
    """
    Сканирование augmented_final
    """
    print("\n" + "="*60)
    print("СКАНИРОВАНИЕ AUGMENTED_FINAL")
    print("="*60)
    
    audio_files = []
    
    # Пути к augmented_final
    base_path = project_root / 'data' / 'processed' / 'augmented_final'
    
    if not base_path.exists():
        print(f"❌ Папка не найдена: {base_path}")
        # Пробуем другие варианты
        alt_paths = [
            project_root / 'data' / 'processed' / 'augmented_8khz',
            project_root / 'data' / 'processed' / 'augmented_8khz_v2'
        ]
        
        for alt in alt_paths:
            if alt.exists():
                base_path = alt
                print(f"✅ Используем: {base_path}")
                break
    
    for category in ['human', 'robot']:
        cat_path = base_path / category
        if cat_path.exists():
            wavs = list(cat_path.rglob('*.wav'))
            print(f"{category}: {len(wavs)} файлов")
            for wav in wavs:
                audio_files.append({
                    'path': wav,
                    'label': 'real' if category == 'human' else 'synthetic'
                })
    
    print(f"\nВсего найдено: {len(audio_files)} файлов")
    return audio_files


def process_file_wrapper(file_info, extractor):
    """Обертка для обработки одного файла"""
    features = extractor.extract_features(file_info['path'])
    if features:
        features['filepath'] = str(file_info['path'])
        features['filename'] = file_info['path'].name
        features['label'] = file_info['label']
        return features
    return None


class FastPhoneticAnalyzer:
    """
    Быстрый анализатор (многопроцессорный)
    """
    
    def __init__(self):
        self.extractor = FastFeatureExtractor(sr=8000)
        self.results = []
        
    def run_analysis(self, output_dir='phonetic_analysis_fast'):
        """
        Запуск многопроцессорного анализа
        """
        output_path = project_root / output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Сканируем файлы
        audio_files = scan_augmented_final()
        
        if not audio_files:
            print("\n❌ Нет файлов для анализа!")
            return
        
        print("\n" + "="*60)
        print(f"МНОГОПРОЦЕССОРНАЯ ОБРАБОТКА (процессов: {CPU_COUNT})")
        print("="*60)
        
        start_time = time.time()
        
        # Используем пул процессов
        with ProcessPoolExecutor(max_workers=CPU_COUNT) as executor:
            # Создаем частичную функцию с фиксированным extractor
            process_func = partial(process_file_wrapper, extractor=self.extractor)
            
            # Запускаем обработку
            futures = [executor.submit(process_func, file_info) for file_info in audio_files]
            
            # Собираем результаты с прогресс-баром
            for future in tqdm(futures, desc="Обработка файлов"):
                try:
                    result = future.result(timeout=10)
                    if result:
                        self.results.append(result)
                except Exception as e:
                    print(f"Ошибка: {e}")
                    continue
        
        total_time = time.time() - start_time
        
        print(f"\n✅ Обработано: {len(self.results)}/{len(audio_files)} файлов")
        print(f"⏱️  Время: {total_time:.1f} сек")
        if len(self.results) > 0:
            print(f"⚡ Скорость: {len(self.results)/total_time:.1f} файлов/сек")
        
        if not self.results:
            return
        
        # Создаем DataFrame
        df = pd.DataFrame(self.results)
        
        # Сохраняем
        csv_path = output_path / 'phonetic_features.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"\n✅ Признаки сохранены в: {csv_path}")
        
        # Статистика
        self.analyze_statistics(df, output_path)
        
        print(f"\n✅ Анализ завершен! Результаты в: {output_path}")
        
    def analyze_statistics(self, df, output_path):
        """
        Статистический анализ
        """
        real_df = df[df['label'] == 'real']
        synth_df = df[df['label'] == 'synthetic']
        
        exclude_cols = ['filepath', 'filename', 'label']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        stats_data = []
        
        for col in feature_cols:
            real_vals = real_df[col].dropna()
            synth_vals = synth_df[col].dropna()
            
            if len(real_vals) == 0 or len(synth_vals) == 0:
                continue
            
            real_mean = float(np.mean(real_vals))
            synth_mean = float(np.mean(synth_vals))
            
            t_stat, p_value = stats.ttest_ind(real_vals, synth_vals, equal_var=False)
            
            pooled_std = float(np.sqrt((np.std(real_vals)**2 + np.std(synth_vals)**2) / 2))
            cohens_d = (real_mean - synth_mean) / pooled_std if pooled_std > 0 else 0
            
            stats_data.append({
                'feature': col,
                'real_mean': real_mean,
                'synthetic_mean': synth_mean,
                'difference': real_mean - synth_mean,
                'diff_percent': ((real_mean - synth_mean) / (synth_mean + 1e-10)) * 100,
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'cohens_d': float(cohens_d)
            })
        
        stats_df = pd.DataFrame(stats_data)
        stats_df = stats_df.sort_values('cohens_d', key=abs, ascending=False)
        
        # Сохраняем
        stats_df.to_csv(output_path / 'statistics.csv', index=False, encoding='utf-8')
        
        # Текстовый отчет
        with open(output_path / 'phonetic_summary.txt', 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("ФОНЕТИЧЕСКИЙ АНАЛИЗ: REAL vs SYNTHETIC\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Всего файлов: {len(real_df) + len(synth_df)}\n")
            f.write(f"  Real: {len(real_df)}\n")
            f.write(f"  Synthetic: {len(synth_df)}\n\n")
            
            f.write("ТОП-10 ПРИЗНАКОВ:\n")
            f.write("-"*60 + "\n")
            
            for i, row in stats_df.head(10).iterrows():
                direction = "выше в REAL" if row['cohens_d'] > 0 else "выше в SYNTHETIC"
                f.write(f"{i+1}. {row['feature']}\n")
                f.write(f"   Real: {row['real_mean']:.4f}\n")
                f.write(f"   Synthetic: {row['synthetic_mean']:.4f}\n")
                f.write(f"   Разница: {abs(row['diff_percent']):.1f}% ({direction})\n")
                f.write(f"   Cohen's d: {abs(row['cohens_d']):.3f}\n")
                f.write(f"   p-value: {row['p_value']:.4f}\n\n")


def main():
    print("="*60)
    print("БЫСТРЫЙ ФОНЕТИЧЕСКИЙ АНАЛИЗ (CPU)")
    print("="*60)
    print(f"Корень проекта: {project_root}")
    print(f"Анализ только: augmented_final")
    print(f"Ядер CPU: {CPU_COUNT}")
    
    analyzer = FastPhoneticAnalyzer()
    analyzer.run_analysis('phonetic_analysis_fast')


if __name__ == "__main__":
    main()