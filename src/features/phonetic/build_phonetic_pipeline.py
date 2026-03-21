#!/usr/bin/env python3
"""
Извлечение фонетических признаков (27) для всех аудиофайлов из разбиения.
Многопроцессорная обработка для ускорения.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import multiprocessing
import json
from datetime import datetime
from functools import partial

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.utils.file_utils import FileManager
from src.utils.logger import setup_logger
from src.utils.config_loader import ConfigLoader
from src.features.phonetic.phonetic_features import PhoneticFeatureExtractor


def parse_args():
    parser = argparse.ArgumentParser(description='Извлечение фонетических признаков')
    parser.add_argument('--config', type=str, default='configs', help='Путь к конфигам')
    parser.add_argument('--workers', type=int, default=None, help='Количество процессов')
    parser.add_argument('--force', action='store_true', help='Принудительное пересоздание')
    return parser.parse_args()


def extract_single_file(file_path, sample_rate=8000):
    """Извлечение признаков для одного файла (вызывается в процессах)"""
    extractor = PhoneticFeatureExtractor(sample_rate=sample_rate)
    features = extractor.extract_all(file_path)
    if features is None:
        return np.zeros(27, dtype=np.float32)
    return features


class PhoneticPipeline:
    def __init__(self, config_path='configs', num_workers=None, force=False):
        self.config_loader = ConfigLoader(config_path)
        self.paths_config = self.config_loader.load_config('paths_config')
        self.file_manager = FileManager()
        self.force = force
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)

        self.processed_root = Path(self.paths_config['paths']['processed_root'])
        self.splits_root = Path(self.paths_config['paths']['splits_root'])
        self.info_file = self.processed_root / 'phonetic_info.json'

        self.sample_rate = 8000

        self.logger = setup_logger('phonetic_builder')
        self.logger.info(f"Workers: {self.num_workers}")
        self.logger.info(f"Выходная директория: {self.processed_root}")

    def check_already_done(self):
        if self.force:
            return False
        required = [
            self.processed_root / 'phonetic_train.npy',
            self.processed_root / 'phonetic_val.npy',
            self.processed_root / 'phonetic_test.npy'
        ]
        return all(f.exists() for f in required)

    def load_file_list(self, split_name):
        split_file = self.splits_root / f'{split_name}_files.txt'
        if not split_file.exists():
            self.logger.error(f"Файл не найден: {split_file}")
            return []
        files = []
        with open(split_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if parts:
                    file_path = parts[0]
                    path = Path(file_path)
                    if not path.exists():
                        self.logger.warning(f"Файл не найден: {file_path}")
                        continue
                    files.append(str(path))
        return files

    def process_split(self, split_name):
        self.logger.info(f"\n--- Обработка split: {split_name} ---")
        file_paths = self.load_file_list(split_name)
        if not file_paths:
            self.logger.error(f"Нет файлов для {split_name}")
            return

        self.logger.info(f"Найдено файлов: {len(file_paths)}")

        # Частичная функция с фиксированным sample_rate
        extract_func = partial(extract_single_file, sample_rate=self.sample_rate)

        # Многопроцессорная обработка
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            features = list(tqdm(
                pool.imap(extract_func, file_paths),
                total=len(file_paths),
                desc=f"{split_name} (параллельно)"
            ))

        features = np.array(features, dtype=np.float32)
        self.logger.info(f"Форма признаков: {features.shape}")
        np.save(self.processed_root / f'phonetic_{split_name}.npy', features)

    def run(self):
        if self.check_already_done():
            self.logger.info("✅ Фонетические признаки уже извлечены, пропускаем")
            return

        self.logger.info("=" * 60)
        self.logger.info("ИЗВЛЕЧЕНИЕ ФОНЕТИЧЕСКИХ ПРИЗНАКОВ (МНОГОПРОЦЕССОРНО)")
        self.logger.info("=" * 60)

        for split in ['train', 'val', 'test']:
            self.process_split(split)

        # Сохраняем информацию
        info = {
            'timestamp': datetime.now().isoformat(),
            'feature_dim': 27,
            'sample_rate': self.sample_rate,
            'num_workers': self.num_workers,
            'force': self.force
        }
        with open(self.info_file, 'w') as f:
            json.dump(info, f, indent=2)

        self.logger.info("\n✅ Готово")


def main():
    args = parse_args()
    pipeline = PhoneticPipeline(
        config_path=args.config,
        num_workers=args.workers,
        force=args.force
    )
    pipeline.run()


if __name__ == "__main__":
    main()