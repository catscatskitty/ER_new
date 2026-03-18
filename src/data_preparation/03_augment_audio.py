"""
Скрипт для аугментации аудиоданных - только шум и телефонный канал
Путь: src/data_preparation/03_augment_audio.py
"""

import argparse
import sys
from pathlib import Path
import random
import json
import shutil
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import concurrent.futures

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.file_utils import FileManager
from src.utils.logger import setup_logger
from src.utils.config_loader import ConfigLoader


def add_noise_to_audio(y, sr, noise_type='white', snr_db=15):
    """
    Добавление шума к аудио
    """
    if noise_type == 'white':
        noise = np.random.normal(0, 1, len(y))
    elif noise_type == 'pink':
        white = np.random.randn(len(y))
        white_fft = np.fft.rfft(white)
        pink_fft = white_fft / np.sqrt(np.arange(1, len(white_fft) + 1))
        noise = np.fft.irfft(pink_fft, len(y))
    elif noise_type == 'street':
        # Уличный шум - имитация равномерным шумом
        noise = np.random.uniform(-1, 1, len(y))
    elif noise_type == 'cafe':
        # Шум кафе - смесь белого и розового
        white = np.random.normal(0, 1, len(y))
        pink = np.random.randn(len(y))
        pink_fft = np.fft.rfft(pink)
        pink_filtered = np.fft.irfft(pink_fft / np.sqrt(np.arange(1, len(pink_fft) + 1)), len(y))
        noise = 0.7 * white + 0.3 * pink_filtered
    else:
        noise = np.random.normal(0, 1, len(y))
    
    signal_power = np.mean(y ** 2)
    noise_power = np.mean(noise ** 2)
    
    snr_linear = 10 ** (snr_db / 10)
    noise_scaled = noise * np.sqrt(signal_power / (snr_linear * noise_power + 1e-10))
    
    return y + noise_scaled


def emulate_phone_channel(y, sr):
    """
    Эмуляция телефонного канала (300-3400 Гц)
    """
    from scipy import signal
    
    nyquist = sr / 2
    low = 300 / nyquist
    high = 3400 / nyquist
    high = min(high, 0.99)
    
    b, a = signal.butter(5, [low, high], btype='band')
    
    y_filtered = signal.filtfilt(b, a, y)
    quantization_noise = np.random.normal(0, 0.005, len(y_filtered))
    y_filtered += quantization_noise
    
    return y_filtered


def augment_single_file(args):
    """
    Аугментация одного файла
    """
    input_path, output_base_dir, class_name, noise_types, snr_range, apply_phone, force, target_sr = args
    
    try:
        # Загружаем аудио
        y, sr = librosa.load(input_path, sr=target_sr)
        
        if y is None or len(y) == 0:
            return None
        
        y = y / (np.max(np.abs(y)) + 1e-10)
        
        # Определяем выходную папку
        if class_name == 'human':
            # Для human сохраняем структуру
            try:
                rel_path = input_path.relative_to(input_path.parent.parent.parent / 'human')
                output_dir = output_base_dir / 'human' / rel_path.parent
            except:
                output_dir = output_base_dir / 'human'
        else:
            # Для robot все в одну папку с префиксом
            output_dir = output_base_dir / 'robot'
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Определяем префикс для robot файлов
        if class_name == 'robot':
            try:
                rel_path = input_path.relative_to(input_path.parent.parent.parent / 'robot')
                if len(rel_path.parts) > 1:
                    prefix = rel_path.parts[0]  # Название подпапки
                else:
                    prefix = 'root'
            except:
                prefix = 'root'
            base_name = f"{prefix}_{input_path.stem}"
        else:
            base_name = input_path.stem
        
        created_files = []
        
        # Проверяем существующие файлы
        if not force:
            existing = list(output_dir.glob(f"{base_name}_*.wav"))
            if existing:
                return [str(f) for f in existing]
        
        # Добавляем шум
        for noise_type in noise_types:
            snr_db = random.uniform(snr_range[0], snr_range[1])
            y_noisy = add_noise_to_audio(y, sr, noise_type, snr_db)
            suffix = f"noise_{noise_type}_snr{snr_db:.0f}"
            output_path = output_dir / f"{base_name}_{suffix}.wav"
            sf.write(output_path, y_noisy, sr)
            created_files.append(str(output_path))
        
        # Эмуляция телефона
        if apply_phone:
            y_phone = emulate_phone_channel(y, sr)
            output_path = output_dir / f"{base_name}_phone.wav"
            sf.write(output_path, y_phone, sr)
            created_files.append(str(output_path))
        
        return created_files
        
    except Exception as e:
        print(f"Ошибка в {input_path}: {e}")
        return None


class AudioAugmenter:
    """Класс для аугментации аудио - только шум и телефон"""
    
    def __init__(self, config_path='configs', force=False, skip_if_exists=True, 
                 batch_size=32, num_workers=None, target_sr=8000):
        
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.load_config('data_config')
        self.file_manager = FileManager()
        self.force = force
        self.skip_if_exists = skip_if_exists
        self.batch_size = batch_size
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)
        self.target_sr = target_sr
        
        paths_config = self.config_loader.load_config('paths_config')
        self.audio_root = Path(paths_config['paths']['audio_root'])
        self.processed_root = Path(paths_config['paths']['processed_root'])
        
        # Папка для аугментированных данных
        self.augmented_dir = self.processed_root / f'augmented_{target_sr//1000}khz'
        
        self.logger = setup_logger('audio_augmenter')
        
        # Параметры аугментации из конфига
        self.aug_config = self.config['data']['augmentation']
        self.noise_types = self.aug_config['noise_types']
        self.snr_range = self.aug_config['snr_range']
        self.apply_phone = self.aug_config['apply_phone_codec']
        
        self.info_file = self.augmented_dir / 'augmentation_info.json'
        
        self.logger.info(f"Целевая частота: {self.target_sr} Гц")
        self.logger.info(f"Типы шума: {self.noise_types}")
        self.logger.info(f"SNR диапазон: {self.snr_range} dB")
        self.logger.info(f"Применять телефонный кодек: {self.apply_phone}")
    
    def collect_files(self):
        """Сбор файлов по классам"""
        human_files = []
        robot_files = []
        
        # Human файлы
        human_dir = self.audio_root / 'human'
        if human_dir.exists():
            for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
                human_files.extend(human_dir.rglob(ext))
        
        # Robot файлы (из всех подпапок)
        robot_dir = self.audio_root / 'robot'
        if robot_dir.exists():
            for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
                robot_files.extend(robot_dir.rglob(ext))
        
        return human_files, robot_files
    
    def check_if_already_done(self):
        """Проверка, выполнялась ли уже аугментация"""
        if self.force:
            self.logger.info("🔄 Принудительный режим")
            return False
        
        if not self.augmented_dir.exists():
            return False
        
        if not self.info_file.exists():
            return False
        
        try:
            with open(self.info_file, 'r', encoding='utf-8') as f:
                info = json.load(f)
            
            timestamp = info.get('timestamp', 'unknown')
            num_files = info.get('total_files_created', 0)
            
            self.logger.info(f"📊 Предыдущая аугментация: {timestamp}")
            self.logger.info(f"   Создано файлов: {num_files}")
            
            if self.skip_if_exists:
                response = input("\n❓ Аугментация уже выполнялась. Выполнить заново? (y/n): ").strip().lower()
                return response != 'y'
            
            return True
            
        except Exception as e:
            return False
    
    def process_files(self, files, class_name):
        """Обработка файлов одного класса"""
        if not files:
            return 0
        
        self.logger.info(f"\nОбработка {class_name}: {len(files)} файлов")
        
        # Подготавливаем аргументы
        args_list = [
            (f, self.augmented_dir, class_name, self.noise_types, 
             self.snr_range, self.apply_phone, self.force, self.target_sr)
            for f in files
        ]
        
        total_created = 0
        
        # Параллельная обработка
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(augment_single_file, args): args[0] for args in args_list}
            
            with tqdm(total=len(args_list), desc=f"Аугментация {class_name}") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        total_created += len(result)
                    pbar.update(1)
        
        return total_created
    
    def run(self):
        """Запуск аугментации"""
        self.logger.info("=" * 60)
        self.logger.info(f"🎧 АУГМЕНТАЦИЯ АУДИОДАННЫХ ({self.target_sr//1000} кГц)")
        self.logger.info(f"Типы шума: {self.noise_types}")
        self.logger.info(f"Телефонный канал: {self.apply_phone}")
        self.logger.info("=" * 60)
        
        if self.check_if_already_done():
            self.logger.info("✅ Аугментация уже выполнена, пропускаем")
            return
        
        # Собираем файлы
        human_files, robot_files = self.collect_files()
        total_original = len(human_files) + len(robot_files)
        
        if total_original == 0:
            self.logger.error("❌ Нет аудиофайлов для обработки")
            return
        
        self.logger.info(f"\n📊 Найдено файлов: Human={len(human_files)}, Robot={len(robot_files)}, Всего={total_original}")
        
        # Подготавливаем директорию
        if self.augmented_dir.exists():
            if self.force:
                shutil.rmtree(self.augmented_dir)
            else:
                backup_dir = self.processed_root / f'augmented_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
                shutil.move(str(self.augmented_dir), str(backup_dir))
        
        self.augmented_dir.mkdir(parents=True, exist_ok=True)
        
        # Обрабатываем файлы
        human_created = self.process_files(human_files, 'human')
        robot_created = self.process_files(robot_files, 'robot')
        total_created = human_created + robot_created
        
        # Сохраняем информацию
        info = {
            'timestamp': datetime.now().isoformat(),
            'target_sr': self.target_sr,
            'original_files': {'human': len(human_files), 'robot': len(robot_files), 'total': total_original},
            'created_files': {'human': human_created, 'robot': robot_created, 'total': total_created},
            'augmentation_params': self.aug_config,
            'force': self.force
        }
        
        with open(self.info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"✅ АУГМЕНТАЦИЯ ЗАВЕРШЕНА")
        self.logger.info(f"   Исходных файлов: {total_original}")
        self.logger.info(f"   Создано файлов: {total_created}")
        self.logger.info(f"   Всего теперь: {total_original + total_created}")
        self.logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Аугментация аудио (шум + телефон)')
    parser.add_argument('--config', type=str, default='configs')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--no-skip', action='store_true')
    parser.add_argument('--sr', type=int, default=8000)
    
    args = parser.parse_args()
    
    augmenter = AudioAugmenter(
        config_path=args.config,
        force=args.force,
        skip_if_exists=not args.no_skip,
        batch_size=args.batch_size,
        num_workers=args.workers,
        target_sr=args.sr
    )
    
    augmenter.run()