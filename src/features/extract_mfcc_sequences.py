"""
Извлечение MFCC-последовательностей (time steps × n_mfcc) для LSTM
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import librosa
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.file_utils import FileManager
from src.utils.logger import setup_logger
from src.utils.config_loader import ConfigLoader


def extract_mfcc_sequence(args):
    """
    Извлечение MFCC последовательности фиксированной длины
    """
    file_path, sample_rate, n_mfcc, hop_length, duration, fixed_time_steps = args
    try:
        y, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
        # Извлекаем MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        # Транспонируем: (time_steps, n_mfcc)
        mfcc = mfcc.T

        # Приводим к фиксированной длине
        time_len = mfcc.shape[0]
        if time_len > fixed_time_steps:
            mfcc = mfcc[:fixed_time_steps, :]
        elif time_len < fixed_time_steps:
            pad = fixed_time_steps - time_len
            mfcc = np.pad(mfcc, ((0, pad), (0, 0)), mode='constant', constant_values=0)

        return mfcc, file_path
    except Exception as e:
        print(f"Ошибка в {file_path}: {e}")
        return None, None


class MFCCExtractor:
    def __init__(self, config_path='configs'):
        self.config_loader = ConfigLoader(config_path)
        self.data_config = self.config_loader.load_config('data_config')
        self.paths_config = self.config_loader.load_config('paths_config')
        self.file_manager = FileManager()

        self.sample_rate = self.data_config['data']['sample_rate']
        self.duration = self.data_config['data']['audio_duration']
        self.n_mfcc = 13
        self.hop_length = 512
        self.fixed_time_steps = 128   # фиксированное количество временных шагов

        self.audio_root = Path(self.paths_config['paths']['audio_root'])
        self.processed_root = Path(self.paths_config['paths']['processed_root'])
        self.splits_root = Path(self.paths_config['paths']['splits_root'])

        self.mfcc_dir = self.processed_root / 'mfcc_sequences'
        self.file_manager.ensure_dir(self.mfcc_dir)

        self.logger = setup_logger('mfcc_extractor')
        self.logger.info(f"Параметры: sample_rate={self.sample_rate}, n_mfcc={self.n_mfcc}, "
                         f"hop_length={self.hop_length}, fixed_time_steps={self.fixed_time_steps}")

    def load_file_list(self, split_name):
        split_file = self.splits_root / f'{split_name}_files.txt'
        if not split_file.exists():
            return [], []
        files, labels = [], []
        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    file_path = parts[0]
                    label = int(parts[1])
                    if Path(file_path).exists():
                        files.append(file_path)
                        labels.append(label)
        return files, labels

    def process_split(self, split_name):
        files, labels = self.load_file_list(split_name)
        if not files:
            self.logger.warning(f"Нет файлов для split {split_name}")
            return

        self.logger.info(f"Обработка split {split_name}: {len(files)} файлов")
        sequences = []
        valid_files = []

        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            args = [(f, self.sample_rate, self.n_mfcc, self.hop_length,
                     self.duration, self.fixed_time_steps) for f in files]
            results = list(tqdm(executor.map(extract_mfcc_sequence, args), total=len(files), desc=split_name))

        file_to_seq = {}
        for seq, file_path in results:
            if seq is not None:
                file_to_seq[file_path] = seq
                valid_files.append(file_path)

        file_to_label = {f: l for f, l in zip(files, labels)}
        valid_labels = [file_to_label[f] for f in valid_files]

        X = np.array([file_to_seq[f] for f in valid_files])
        y = np.array(valid_labels)

        np.save(self.mfcc_dir / f'mfcc_{split_name}.npy', X)
        np.save(self.mfcc_dir / f'labels_{split_name}.npy', y)
        self.logger.info(f"Сохранены MFCC-последовательности для {split_name}: {X.shape}")

    def run(self):
        for split in ['train', 'val', 'test']:
            self.process_split(split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs')
    args = parser.parse_args()
    extractor = MFCCExtractor(config_path=args.config)
    extractor.run()