import os
import librosa
import soundfile as sf
from pathlib import Path
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_file(args):
    """
    Конвертация одного файла в 8 кГц
    """
    input_path, output_path = args
    
    try:
        # Загрузка аудио с оригинальной частотой
        audio, sr = librosa.load(input_path, sr=None, mono=True)
        
        # Ресемплинг до 8 кГц
        if sr != 8000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=8000)
        
        # Обрезка/дополнение до 5 секунд
        target_length = 5 * 8000
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            import numpy as np
            audio = np.pad(audio, (0, target_length - len(audio)))
        
        # Сохранение
        sf.write(output_path, audio, 8000)
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при конвертации {input_path}: {e}")
        return False

def convert_all():
    """
    Конвертация всех аудиофайлов в 8 кГц
    """
    # Входные директории
    input_dirs = [
        Path('data/audio/human'),
        Path('data/audio/robot'),
        Path('data/audio/robot/rhvoice')
    ]
    
    # Выходные директории
    output_base = Path('data/processed/augmented_8khz')
    
    # Собираем все файлы для конвертации
    convert_tasks = []
    
    for input_dir in input_dirs:
        if not input_dir.exists():
            continue
            
        # Определяем выходную директорию
        if 'human' in str(input_dir):
            output_dir = output_base / 'human'
        else:
            output_dir = output_base / 'robot' / input_dir.parent.name
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Добавляем задачи
        for audio_file in input_dir.glob('*.wav'):
            output_file = output_dir / audio_file.name
            if not output_file.exists():  # Пропускаем если уже есть
                convert_tasks.append((str(audio_file), str(output_file)))
    
    logger.info(f"Найдено {len(convert_tasks)} файлов для конвертации")
    
    if not convert_tasks:
        logger.info("Новых файлов для конвертации нет")
        return
    
    # Параллельная конвертация
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(executor.map(convert_file, convert_tasks), 
                           total=len(convert_tasks), 
                           desc="Конвертация в 8 кГц"))
    
    success_count = sum(results)
    logger.info(f"Конвертация завершена. Успешно: {success_count}/{len(convert_tasks)}")

if __name__ == "__main__":
    convert_all()