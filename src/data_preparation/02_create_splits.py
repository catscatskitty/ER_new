#!/usr/bin/env python
"""
Шаг 2: Создание разбиения train/val/test
- Сохранение списков файлов в data/splits/
- Пути сохраняются относительно data/processed/augmented_8khz
"""

import numpy as np
from pathlib import Path
import json
import random
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class SplitCreator:
    def __init__(self, audio_dir='data/processed/augmented_8khz', 
                 splits_dir='data/splits',
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                 seed=42):
        self.audio_dir = Path(audio_dir)
        self.splits_dir = Path(splits_dir)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.splits = {
            'train': [],
            'val': [],
            'test': []
        }
        
        self.statistics = {
            'total_files': 0,
            'human': 0,
            'robot': 0,
            'train': {'files': 0, 'human': 0, 'robot': 0},
            'val': {'files': 0, 'human': 0, 'robot': 0},
            'test': {'files': 0, 'human': 0, 'robot': 0}
        }
    
    def collect_files(self):
        """Сбор всех аудиофайлов с метками"""
        print("📂 Collecting audio files...")
        
        self.files_by_class = defaultdict(list)
        
        for class_name in ['human', 'robot']:
            class_dir = self.audio_dir / class_name
            if not class_dir.exists():
                print(f"⚠️ Directory not found: {class_dir}")
                continue
            
            for audio_file in class_dir.glob('*.wav'):
                # Сохраняем путь относительно audio_dir
                # Например: "human/8f9597202b59d0ef2c2ca869bd7303ac_noise_pink_snr10.wav"
                rel_path = audio_file.relative_to(self.audio_dir)
                self.files_by_class[class_name].append(str(rel_path))
                self.statistics[class_name] += 1
        
        self.statistics['total_files'] = sum(self.statistics[class_name] for class_name in ['human', 'robot'])
        
        print(f"  Human: {self.statistics['human']} files")
        print(f"  Robot: {self.statistics['robot']} files")
        print(f"  Total: {self.statistics['total_files']} files")
    
    def create_splits(self):
        """Создание стратифицированного разбиения"""
        print("\n📊 Creating train/val/test splits...")
        
        for class_name, files in self.files_by_class.items():
            n_files = len(files)
            n_train = int(n_files * self.train_ratio)
            n_val = int(n_files * self.val_ratio)
            n_test = n_files - n_train - n_val
            
            # Перемешивание
            shuffled = files.copy()
            random.shuffle(shuffled)
            
            train_files = shuffled[:n_train]
            val_files = shuffled[n_train:n_train + n_val]
            test_files = shuffled[n_train + n_val:]
            
            self.splits['train'].extend(train_files)
            self.splits['val'].extend(val_files)
            self.splits['test'].extend(test_files)
            
            self.statistics['train'][class_name] = len(train_files)
            self.statistics['val'][class_name] = len(val_files)
            self.statistics['test'][class_name] = len(test_files)
        
        # Перемешиваем финальные списки
        for split in self.splits:
            random.shuffle(self.splits[split])
        
        self.statistics['train']['files'] = len(self.splits['train'])
        self.statistics['val']['files'] = len(self.splits['val'])
        self.statistics['test']['files'] = len(self.splits['test'])
        
        print(f"  Train: {self.statistics['train']['files']} files")
        print(f"    Human: {self.statistics['train']['human']}, Robot: {self.statistics['train']['robot']}")
        print(f"  Val:   {self.statistics['val']['files']} files")
        print(f"    Human: {self.statistics['val']['human']}, Robot: {self.statistics['val']['robot']}")
        print(f"  Test:  {self.statistics['test']['files']} files")
        print(f"    Human: {self.statistics['test']['human']}, Robot: {self.statistics['test']['robot']}")
    
    def save_splits(self):
        """Сохранение разбиения в файлы"""
        print("\n💾 Saving splits...")
        
        for split_name, files in self.splits.items():
            split_file = self.splits_dir / f"{split_name}_files.txt"
            with open(split_file, 'w', encoding='utf-8') as f:
                for file_path in files:
                    f.write(f"{file_path}\n")
            print(f"  Saved {len(files)} files to {split_file}")
        
        # Сохранение статистики
        stats_file = self.splits_dir / 'split_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.statistics, f, indent=2, ensure_ascii=False)
        print(f"  Saved statistics to {stats_file}")
        
        # Сохранение конфигурации разбиения
        config = {
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'seed': 42,
            'stratified': True,
            'audio_dir': str(self.audio_dir)
        }
        config_file = self.splits_dir / 'split_config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print(f"  Saved config to {config_file}")
        
        # Вывод примера путей
        print("\n📁 Example paths (first 3 files):")
        for split_name in ['train', 'val', 'test']:
            if self.splits[split_name]:
                print(f"  {split_name}: {self.splits[split_name][0]}")
    
    def run(self):
        """Запуск создания разбиения"""
        print("="*60)
        print("Split Creator")
        print("="*60)
        print(f"Audio directory: {self.audio_dir}")
        print(f"Splits directory: {self.splits_dir}")
        print(f"Split ratios: train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio}")
        print("="*60)
        
        self.collect_files()
        self.create_splits()
        self.save_splits()
        
        print("\n✅ Splits created successfully!")
        
        # Вывод итогов
        print("\n📊 Final Statistics:")
        print(f"  Train: {self.statistics['train']['files']} (human: {self.statistics['train']['human']}, robot: {self.statistics['train']['robot']})")
        print(f"  Val:   {self.statistics['val']['files']} (human: {self.statistics['val']['human']}, robot: {self.statistics['val']['robot']})")
        print(f"  Test:  {self.statistics['test']['files']} (human: {self.statistics['test']['human']}, robot: {self.statistics['test']['robot']})")


def main():
    creator = SplitCreator(
        audio_dir='data/processed/augmented_8khz',
        splits_dir='data/splits',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    creator.run()


if __name__ == "__main__":
    main()