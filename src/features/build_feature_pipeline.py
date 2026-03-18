"""
Извлечение 38 признаков из аудио
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

FEATURE_DIM = 38


def extract_features(file_path, sample_rate=8000, max_duration=5):
    """Извлечение 38 признаков из одного файла"""
    try:
        y, sr = librosa.load(file_path, sr=sample_rate, duration=max_duration)
        
        if y is None or len(y) == 0:
            return None
        
        y = y / (np.max(np.abs(y)) + 1e-10)
        
        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        idx = 0
        
        # 1. MFCC (13 means + 13 stds = 26)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=256)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        features[idx:idx+13] = mfcc_mean[:13]
        idx += 13
        features[idx:idx+13] = mfcc_std[:13]
        idx += 13
        
        # 2. Спектральные признаки (3)
        try:
            features[idx] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=512, hop_length=256))
        except:
            features[idx] = 0
        idx += 1
        
        try:
            features[idx] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=512, hop_length=256))
        except:
            features[idx] = 0
        idx += 1
        
        try:
            features[idx] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=512, hop_length=256))
        except:
            features[idx] = 0
        idx += 1
        
        # 3. ZCR (1)
        try:
            features[idx] = np.mean(librosa.feature.zero_crossing_rate(y, frame_length=512, hop_length=256))
        except:
            features[idx] = 0
        idx += 1
        
        # 4. RMS (1)
        try:
            features[idx] = np.mean(librosa.feature.rms(y=y, frame_length=512, hop_length=256))
        except:
            features[idx] = 0
        idx += 1
        
        # 5. Tempo (1)
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=256)
            if isinstance(tempo, np.ndarray):
                tempo = tempo[0] if len(tempo) > 0 else 0
            features[idx] = float(tempo)
        except:
            features[idx] = 0
        idx += 1
        
        # 6. Chroma (6 признаков)
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=512, hop_length=256)
            chroma_mean = np.mean(chroma, axis=1)
            if len(chroma_mean) >= 6:
                features[idx:idx+6] = chroma_mean[:6]
            else:
                features[idx:idx+len(chroma_mean)] = chroma_mean
        except:
            pass
        idx += 6
        
        return features
        
    except Exception as e:
        return None


def process_file_wrapper(args):
    file_path, label, sample_rate, max_duration = args
    features = extract_features(file_path, sample_rate, max_duration)
    if features is not None:
        return str(file_path), label, features
    return None, None, None


class FeatureBuilder:
    def __init__(self, config_path='configs', num_workers=None, force=False):
        self.config_loader = ConfigLoader(config_path)
        self.data_config = self.config_loader.load_config('data_config')
        self.file_manager = FileManager()
        
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)
        self.force = force
        self.sample_rate = 8000
        self.max_duration = self.data_config['data']['audio_duration']
        
        paths_config = self.config_loader.load_config('paths_config')
        self.processed_root = Path(paths_config['paths']['processed_root'])
        self.splits_root = Path(paths_config['paths']['splits_root'])
        
        self.file_manager.ensure_dir(self.processed_root)
        self.logger = setup_logger('feature_builder')
        
        self.info_file = self.processed_root / 'features_info.json'
    
    def check_if_already_done(self):
        if self.force:
            return False
        required = [self.processed_root / f'features_{s}.npy' for s in ['train', 'val', 'test']]
        required += [self.processed_root / f'labels_{s}.npy' for s in ['train', 'val', 'test']]
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
        
        args_list = [(f, l, self.sample_rate, self.max_duration) for f, l in zip(files, labels)]
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
        
        np.save(self.processed_root / f'features_{split_name}.npy', X)
        np.save(self.processed_root / f'labels_{split_name}.npy', y)
        
        self.logger.info(f"✅ {split_name}: {X.shape}")
        return X, y
    
    def run(self):
        self.logger.info("=" * 60)
        self.logger.info("ИЗВЛЕЧЕНИЕ 38 ПРИЗНАКОВ")
        self.logger.info("=" * 60)
        
        if self.check_if_already_done():
            self.logger.info("✅ Признаки уже извлечены")
            return
        
        for split in ['train', 'val', 'test']:
            self.process_split(split)
        
        info = {
            'timestamp': datetime.now().isoformat(),
            'feature_dim': FEATURE_DIM,
            'sample_rate': self.sample_rate
        }
        with open(self.info_file, 'w') as f:
            json.dump(info, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    
    builder = FeatureBuilder(args.config, args.workers, args.force)
    builder.run()