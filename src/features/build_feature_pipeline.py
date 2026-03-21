"""
Извлечение признаков (акустических + фонетических)
Путь: src/features/build_feature_pipeline.py
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import multiprocessing
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.file_utils import FileManager
from src.utils.logger import setup_logger
from src.utils.config_loader import ConfigLoader
from src.features.phonetic_features import CombinedFeatureExtractor

# Размерности признаков
ACOUSTIC_DIM = 38
PHONETIC_DIM = 27
COMBINED_DIM = ACOUSTIC_DIM + PHONETIC_DIM  # 65


def extract_combined_features(file_path, extractor):
    """Извлечение объединенных признаков"""
    return extractor.extract_combined(file_path)


def extract_acoustic_only(file_path, sample_rate=8000, max_duration=5):
    """Извлечение только акустических признаков (для обратной совместимости)"""
    try:
        y, sr = librosa.load(file_path, sr=sample_rate, duration=max_duration)
        
        if y is None or len(y) == 0:
            return None
        
        y = y / (np.max(np.abs(y)) + 1e-10)
        
        features = []
        
        # MFCC (26)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=256)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        features.extend(mfcc_mean)
        features.extend(mfcc_std)
        
        # Спектральные (3)
        try:
            features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=512, hop_length=256)))
        except:
            features.append(0)
        
        try:
            features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=512, hop_length=256)))
        except:
            features.append(0)
        
        try:
            features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=512, hop_length=256)))
        except:
            features.append(0)
        
        # ZCR (1)
        try:
            features.append(np.mean(librosa.feature.zero_crossing_rate(y, frame_length=512, hop_length=256)))
        except:
            features.append(0)
        
        # RMS (1)
        try:
            features.append(np.mean(librosa.feature.rms(y=y, frame_length=512, hop_length=256)))
        except:
            features.append(0)
        
        # Tempo (1)
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=256)
            if isinstance(tempo, np.ndarray):
                tempo = tempo[0] if len(tempo) > 0 else 0
            features.append(float(tempo))
        except:
            features.append(0)
        
        # Chroma (6)
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=512, hop_length=256)
            chroma_mean = np.mean(chroma, axis=1)
            features.extend(chroma_mean[:6])
        except:
            features.extend([0, 0, 0, 0, 0, 0])
        
        if len(features) != 38:
            if len(features) < 38:
                features.extend([0] * (38 - len(features)))
            else:
                features = features[:38]
        
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        return None


def process_file_wrapper(args):
    """Обертка для обработки одного файла"""
    file_path, label, feature_type, extractor = args
    
    if feature_type == 'combined':
        features = extract_combined_features(file_path, extractor)
    else:
        features = extract_acoustic_only(file_path)
    
    if features is not None:
        return str(file_path), label, features
    return None, None, None


class FeatureBuilder:
    def __init__(self, config_path='configs', num_workers=None, force=False, feature_type='acoustic'):
        self.config_loader = ConfigLoader(config_path)
        self.data_config = self.config_loader.load_config('data_config')
        self.file_manager = FileManager()
        
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)
        self.force = force
        self.feature_type = feature_type
        self.sample_rate = 8000
        self.max_duration = self.data_config['data']['audio_duration']
        
        # Определяем размерность
        if feature_type == 'combined':
            self.feature_dim = COMBINED_DIM
            self.extractor = CombinedFeatureExtractor()
        else:
            self.feature_dim = ACOUSTIC_DIM
            self.extractor = None
        
        paths_config = self.config_loader.load_config('paths_config')
        self.processed_root = Path(paths_config['paths']['processed_root'])
        self.splits_root = Path(paths_config['paths']['splits_root'])
        
        self.file_manager.ensure_dir(self.processed_root)
        self.logger = setup_logger('feature_builder')
        
        self.info_file = self.processed_root / 'features_info.json'
        
        self.logger.info("=" * 60)
        self.logger.info(f"ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ ({feature_type})")
        self.logger.info(f"Размерность: {self.feature_dim}")
        self.logger.info("=" * 60)
    
    def check_if_already_done(self):
        if self.force:
            return False
        
        suffix = f"_{self.feature_type}" if self.feature_type != 'acoustic' else ''
        required = [
            self.processed_root / f'features_{s}{suffix}.npy' for s in ['train', 'val', 'test']
        ]
        required += [
            self.processed_root / f'labels_{s}.npy' for s in ['train', 'val', 'test']
        ]
        
        return all(f.exists() for f in required)
    
    def load_file_list(self, split_name):
        split_file = self.splits_root / f'{split_name}_files.txt'
        if not split_file.exists():
            return [], []
        
        files, labels = [], []
        with open(split_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    path = Path(parts[0])
                    if path.exists():
                        files.append(str(path))
                        labels.append(int(parts[1]))
        return files, labels
    
    def process_split(self, split_name):
        self.logger.info(f"\nОбработка {split_name}")
        files, labels = self.load_file_list(split_name)
        
        if not files:
            self.logger.error(f"Нет файлов для {split_name}")
            return None, None
        
        args_list = [(f, l, self.feature_type, self.extractor) for f, l in zip(files, labels)]
        all_features, all_labels, all_files = [], [], []
        
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            with tqdm(total=len(args_list), desc=split_name) as pbar:
                for result in pool.imap_unordered(process_file_wrapper, args_list):
                    f, l, feat = result
                    if feat is not None:
                        all_features.append(feat)
                        all_labels.append(l)
                        all_files.append(f)
                    pbar.update(1)
        
        if not all_features:
            return None, None
        
        X = np.array(all_features, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int32)
        
        suffix = f"_{self.feature_type}" if self.feature_type != 'acoustic' else ''
        np.save(self.processed_root / f'features_{split_name}{suffix}.npy', X)
        np.save(self.processed_root / f'labels_{split_name}.npy', y)
        
        self.logger.info(f"✅ {split_name}: {X.shape}")
        return X, y
    
    def run(self):
        if self.check_if_already_done():
            self.logger.info("✅ Признаки уже извлечены")
            return
        
        for split in ['train', 'val', 'test']:
            self.process_split(split)
        
        info = {
            'timestamp': datetime.now().isoformat(),
            'feature_type': self.feature_type,
            'feature_dim': self.feature_dim,
            'sample_rate': self.sample_rate
        }
        with open(self.info_file, 'w') as f:
            json.dump(info, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--features', type=str, default='acoustic', 
                       choices=['acoustic', 'combined'],
                       help='Тип признаков: acoustic или combined (акустика+фонетика)')
    args = parser.parse_args()
    
    builder = FeatureBuilder(args.config, args.workers, args.force, args.features)
    builder.run()