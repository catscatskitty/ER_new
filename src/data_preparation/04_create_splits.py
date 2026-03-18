"""
Скрипт для разбиения данных на train/val/test с учетом аугментации
Путь: src/data_preparation/04_create_splits.py
"""

import argparse
import sys
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
import json

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.file_utils import FileManager
from src.utils.logger import setup_logger
from src.utils.config_loader import ConfigLoader


class DataSplitter:
    def __init__(self, config_path='configs'):
        self.config_loader = ConfigLoader(config_path)
        self.file_manager = FileManager()
        
        try:
            self.training_config = self.config_loader.load_config('training_config')
            self.paths_config = self.config_loader.load_config('paths_config')
        except:
            self.training_config = {'training': {}}
            self.paths_config = {'paths': {}}
        
        self.processed_root = Path(self.paths_config['paths'].get('processed_root', './data/processed'))
        self.splits_root = Path(self.paths_config['paths'].get('splits_root', './data/splits'))
        self.audio_root = Path(self.paths_config['paths'].get('audio_root', './data/audio'))
        
        self.file_manager.ensure_dir(self.splits_root)
        self.logger = setup_logger('data_splitter')
        
        training_params = self.training_config.get('training', {})
        self.train_ratio = training_params.get('train_ratio', 0.8)
        self.val_ratio = training_params.get('val_ratio', 0.1)
        self.random_seed = training_params.get('random_seed', 42)
        
        random.seed(self.random_seed)
    
    def collect_all_files(self):
        """Сбор ВСЕХ файлов: оригинальные + аугментированные"""
        all_files = []
        
        # 1. Оригинальные файлы
        # Human
        human_original = list((self.audio_root / 'human').rglob('*.wav'))
        for f in human_original:
            all_files.append((str(f), 0, 'original_human'))
        
        # Robot
        robot_original = list((self.audio_root / 'robot').rglob('*.wav'))
        for f in robot_original:
            all_files.append((str(f), 1, 'original_robot'))
        
        # 2. Аугментированные файлы
        augmented_dirs = list(self.processed_root.glob('augmented_*khz'))
        for aug_dir in augmented_dirs:
            # Human аугментированные
            human_aug = list((aug_dir / 'human').rglob('*.wav'))
            for f in human_aug:
                all_files.append((str(f), 0, 'augmented_human'))
            
            # Robot аугментированные
            robot_aug = list((aug_dir / 'robot').rglob('*.wav'))
            for f in robot_aug:
                all_files.append((str(f), 1, 'augmented_robot'))
        
        return all_files
    
    def create_splits(self):
        self.logger.info("=" * 60)
        self.logger.info("СОЗДАНИЕ РАЗБИЕНИЯ ДАННЫХ")
        self.logger.info("=" * 60)
        
        all_files = self.collect_all_files()
        
        if len(all_files) == 0:
            self.logger.error("❌ Нет файлов для разбиения")
            return
        
        # Разделяем по меткам
        human_files = [f for f in all_files if f[1] == 0]
        robot_files = [f for f in all_files if f[1] == 1]
        
        random.shuffle(human_files)
        random.shuffle(robot_files)
        
        # Разбиваем human
        human_train, human_temp = train_test_split(
            human_files, train_size=self.train_ratio, random_state=self.random_seed
        )
        human_val, human_test = train_test_split(
            human_temp, train_size=0.5, random_state=self.random_seed
        )
        
        # Разбиваем robot
        robot_train, robot_temp = train_test_split(
            robot_files, train_size=self.train_ratio, random_state=self.random_seed
        )
        robot_val, robot_test = train_test_split(
            robot_temp, train_size=0.5, random_state=self.random_seed
        )
        
        # Объединяем
        train_files = human_train + robot_train
        val_files = human_val + robot_val
        test_files = human_test + robot_test
        
        random.shuffle(train_files)
        random.shuffle(val_files)
        random.shuffle(test_files)
        
        # Сохраняем
        for name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            with open(self.splits_root / f'{name}_files.txt', 'w', encoding='utf-8') as f:
                for file_path, label, source in files:
                    f.write(f"{file_path}\t{label}\n")
        
        # Статистика
        self.logger.info(f"\n📊 Статистика:")
        self.logger.info(f"  Train: {len(train_files)} (Human: {len(human_train)}, Robot: {len(robot_train)})")
        self.logger.info(f"  Val: {len(val_files)} (Human: {len(human_val)}, Robot: {len(robot_val)})")
        self.logger.info(f"  Test: {len(test_files)} (Human: {len(human_test)}, Robot: {len(robot_test)})")
        
        stats = {
            'train': len(train_files),
            'val': len(val_files),
            'test': len(test_files),
            'total': len(all_files)
        }
        
        with open(self.splits_root / 'split_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"\n✅ Разбиение сохранено в {self.splits_root}")

    def run(self):
        self.create_splits()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Разбиение данных')
    parser.add_argument('--config', type=str, default='configs')
    args = parser.parse_args()
    
    splitter = DataSplitter(args.config)
    splitter.run()