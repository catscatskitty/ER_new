#!/usr/bin/env python3
"""
Скрипт для копирования всех тестовых аудиофайлов в одну папку
Путь: scripts/collect_test_files.py
"""

import argparse
import sys
from pathlib import Path
import shutil
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.file_utils import FileManager
from src.utils.logger import setup_logger
from src.utils.config_loader import ConfigLoader


class TestFileCollector:
    """Класс для сбора всех тестовых файлов в одну папку"""
    
    def __init__(self, config_path='configs', output_dir='collected_test_files'):
        self.config_loader = ConfigLoader(config_path)
        self.file_manager = FileManager()
        self.logger = setup_logger('test_collector')
        
        # Загружаем пути
        try:
            paths_config = self.config_loader.load_config('paths_config')
            self.splits_root = Path(paths_config['paths']['splits_root'])
            self.audio_root = Path(paths_config['paths']['audio_root'])
            self.processed_root = Path(paths_config['paths']['processed_root'])
        except Exception as e:
            self.logger.warning(f"Ошибка загрузки конфига: {e}")
            self.splits_root = Path('data/splits')
            self.audio_root = Path('data/audio')
            self.processed_root = Path('data/processed')
        
        # Выходная папка
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Создаем подпапки для разных типов
        self.human_dir = self.output_dir / 'human'
        self.robot_dir = self.output_dir / 'robot'
        self.human_dir.mkdir(exist_ok=True)
        self.robot_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"📁 Файлы будут сохранены в: {self.output_dir.absolute()}")
    
    def collect_from_split_file(self, split_name='test'):
        """Сбор файлов из указанного split файла"""
        split_file = self.splits_root / f'{split_name}_files.txt'
        
        if not split_file.exists():
            self.logger.error(f"❌ Файл не найден: {split_file}")
            return 0, 0
        
        self.logger.info(f"📄 Загрузка списка файлов из {split_file}")
        
        files_to_copy = []
        
        with open(split_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 1:
                    file_path = parts[0]
                    label = int(parts[1]) if len(parts) > 1 else None
                    files_to_copy.append((file_path, label))
        
        self.logger.info(f"Найдено {len(files_to_copy)} файлов в {split_name} наборе")
        
        # Копируем файлы
        human_copied = 0
        robot_copied = 0
        
        for file_path, label in tqdm(files_to_copy, desc="Копирование файлов"):
            src = Path(file_path)
            
            if not src.exists():
                # Пробуем найти в других местах
                alt_paths = [
                    self.audio_root / src.name,
                    self.processed_root / 'augmented_8khz' / src.name,
                    self.processed_root / 'augmented_8khz' / 'human' / src.name,
                    self.processed_root / 'augmented_8khz' / 'robot' / src.name,
                ]
                
                found = False
                for alt in alt_paths:
                    if alt.exists():
                        src = alt
                        found = True
                        break
                
                if not found:
                    self.logger.warning(f"⚠️ Файл не найден: {file_path}")
                    continue
            
            # Определяем целевую папку по метке или по исходному пути
            if label == 0:
                dst = self.human_dir / src.name
                human_copied += 1
            elif label == 1:
                dst = self.robot_dir / src.name
                robot_copied += 1
            else:
                # Если метки нет, определяем по пути
                if 'human' in str(src).lower():
                    dst = self.human_dir / src.name
                    human_copied += 1
                else:
                    dst = self.robot_dir / src.name
                    robot_copied += 1
            
            # Копируем файл
            shutil.copy2(src, dst)
        
        self.logger.info(f"✅ Скопировано: human={human_copied}, robot={robot_copied}")
        return human_copied, robot_copied
    
    def collect_from_directories(self, max_files_per_class=None):
        """Сбор файлов напрямую из директорий (без split файла)"""
        self.logger.info("📁 Сбор файлов из директорий")
        
        # Собираем human файлы
        human_sources = [
            self.audio_root / 'human',
            self.processed_root / 'augmented_8khz' / 'human',
        ]
        
        robot_sources = [
            self.audio_root / 'robot',
            self.processed_root / 'augmented_8khz' / 'robot',
        ]
        
        human_files = []
        robot_files = []
        
        # Собираем human
        for src_dir in human_sources:
            if src_dir.exists():
                files = list(src_dir.rglob('*.wav'))
                human_files.extend(files)
                self.logger.info(f"  Найдено human в {src_dir}: {len(files)}")
        
        # Собираем robot
        for src_dir in robot_sources:
            if src_dir.exists():
                files = list(src_dir.rglob('*.wav'))
                robot_files.extend(files)
                self.logger.info(f"  Найдено robot в {src_dir}: {len(files)}")
        
        # Удаляем дубликаты
        human_files = list(set(human_files))
        robot_files = list(set(robot_files))
        
        self.logger.info(f"Всего уникальных: human={len(human_files)}, robot={len(robot_files)}")
        
        # Ограничиваем количество если нужно
        if max_files_per_class:
            import random
            random.shuffle(human_files)
            random.shuffle(robot_files)
            human_files = human_files[:max_files_per_class]
            robot_files = robot_files[:max_files_per_class]
            self.logger.info(f"После ограничения: human={len(human_files)}, robot={len(robot_files)}")
        
        # Копируем файлы
        human_copied = 0
        robot_copied = 0
        
        self.logger.info("Копирование human файлов...")
        for src in tqdm(human_files, desc="Human"):
            dst = self.human_dir / src.name
            # Если файл с таким именем уже есть, добавляем префикс
            if dst.exists():
                dst = self.human_dir / f"human_{human_copied}_{src.name}"
            shutil.copy2(src, dst)
            human_copied += 1
        
        self.logger.info("Копирование robot файлов...")
        for src in tqdm(robot_files, desc="Robot"):
            dst = self.robot_dir / src.name
            if dst.exists():
                dst = self.robot_dir / f"robot_{robot_copied}_{src.name}"
            shutil.copy2(src, dst)
            robot_copied += 1
        
        self.logger.info(f"✅ Скопировано: human={human_copied}, robot={robot_copied}")
        return human_copied, robot_copied
    
    def create_metadata(self):
        """Создание metadata.csv с информацией о собранных файлах"""
        import csv
        
        metadata = []
        
        # Human файлы
        for f in sorted(self.human_dir.glob('*.wav')):
            metadata.append({
                'filename': f.name,
                'path': str(f.relative_to(self.output_dir)),
                'label': 'human',
                'class': 0
            })
        
        # Robot файлы
        for f in sorted(self.robot_dir.glob('*.wav')):
            metadata.append({
                'filename': f.name,
                'path': str(f.relative_to(self.output_dir)),
                'label': 'robot',
                'class': 1
            })
        
        # Сохраняем CSV
        import pandas as pd
        df = pd.DataFrame(metadata)
        csv_path = self.output_dir / 'metadata.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        self.logger.info(f"📊 Метаданные сохранены в {csv_path}")
        return df
    
    def run(self, method='split', split_name='test', max_files=None):
        """
        Запуск сбора файлов
        
        Args:
            method: 'split' - из split файла, 'direct' - из директорий
            split_name: имя split файла (test/val/train)
            max_files: максимальное количество файлов на класс (для direct метода)
        """
        self.logger.info("=" * 60)
        self.logger.info("📂 СБОР ТЕСТОВЫХ ФАЙЛОВ В ОДНУ ПАПКУ")
        self.logger.info("=" * 60)
        
        if method == 'split':
            human, robot = self.collect_from_split_file(split_name)
        else:
            human, robot = self.collect_from_directories(max_files)
        
        if human + robot == 0:
            self.logger.error("❌ Не найдено файлов для копирования")
            return
        
        self.logger.info(f"\n📊 ИТОГИ:")
        self.logger.info(f"  Human файлов: {human}")
        self.logger.info(f"  Robot файлов: {robot}")
        self.logger.info(f"  Всего файлов: {human + robot}")
        self.logger.info(f"  Директория: {self.output_dir.absolute()}")
        
        # Создаем метаданные
        self.create_metadata()
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("✅ СБОР ЗАВЕРШЕН")
        self.logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Сбор тестовых файлов в одну папку')
    parser.add_argument('--config', type=str, default='configs', help='Путь к конфигам')
    parser.add_argument('--output', type=str, default='collected_test_files', help='Выходная папка')
    parser.add_argument('--method', type=str, choices=['split', 'direct'], default='split',
                       help='Метод сбора: split - из split файла, direct - из директорий')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                       help='Имя split файла (для метода split)')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Максимальное количество файлов на класс (для метода direct)')
    
    args = parser.parse_args()
    
    collector = TestFileCollector(config_path=args.config, output_dir=args.output)
    collector.run(method=args.method, split_name=args.split, max_files=args.max_files)


if __name__ == "__main__":
    main()