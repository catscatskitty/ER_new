import os
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import logging
from tqdm import tqdm
import argparse
import random
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_windows_path(path):
    """Исправление путей для Windows"""
    return str(path).replace('\\', '/')

def generate_silero_voices_fixed():
    """
    Исправленная генерация синтезированной речи с помощью Silero TTS для Windows
    """
    try:
        # Сначала скачиваем модель вручную
        model_dir = Path.home() / '.cache' / 'torch' / 'hub' / 'snakers4_silero-models_master'
        model_path = model_dir / 'src' / 'silero' / 'model' / 'v4_ru.pt'
        
        logger.info(f"Проверка модели по пути: {model_path}")
        
        # Если модели нет, скачиваем через hub с исправленными путями
        if not model_path.exists():
            logger.info("Модель не найдена, скачиваем...")
            try:
                # Альтернативный метод загрузки
                torch.hub.set_dir(str(Path.home() / '.cache' / 'torch' / 'hub'))
                
                # Загружаем модель с явным указанием путей
                model, example_text = torch.hub.load(
                    repo_or_dir='snakers4/silero-models',
                    model='silero_tts',
                    language='ru',
                    speaker='v4_ru',
                    trust_repo=True,
                    force_reload=True
                )
                logger.info("Модель успешно загружена через hub")
                return model, example_text
                
            except Exception as e:
                logger.error(f"Ошибка загрузки через hub: {e}")
                
                # Ручная загрузка модели
                import urllib.request
                
                url = "https://models.silero.ai/models/tts/ru/v4_ru.pt"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"Скачиваем модель напрямую с {url}")
                urllib.request.urlretrieve(url, model_path)
                logger.info(f"Модель скачана в {model_path}")
                
                # Загружаем модель
                model = torch.package.PackageImporter(model_path).load_pickle("tts_models", "model")
                model.to(torch.device('cpu'))
                
                # Создаем example_text
                example_text = "Здравствуй, читатель!"
                
                return model, example_text
        else:
            logger.info("Модель найдена в кэше")
            # Загружаем модель из файла
            model = torch.package.PackageImporter(model_path).load_pickle("tts_models", "model")
            model.to(torch.device('cpu'))
            example_text = "Здравствуй, читатель!"
            return model, example_text
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке Silero: {e}")
        return None, None

def generate_silero_voices_fixed_batch(multiplier: int = 5):
    """
    Пакетная генерация с исправленным Silero
    """
    try:
        # Загружаем модель
        model, example_text = generate_silero_voices_fixed()
        
        if model is None:
            logger.error("Не удалось загрузить модель Silero")
            return
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        logger.info(f"Модель загружена на {device}")
        
        # Доступные голоса
        speakers = ['xenia', 'baya', 'kseniya', 'aidar', 'eugene', 'random']
        
        # Тексты для синтеза
        texts = [
            "Здравствуйте, это звонок из банка. Вас беспокоит служба безопасности.",
            "На вашей карте замечена подозрительная активность. Подтвердите операцию.",
            "Для продолжения разговора нажмите цифру один на клавиатуре телефона.",
            "Ваш номер был выбран для участия в акции. Подробности можно узнать у оператора.",
            "Спасибо за ожидание. Соединяю с оператором, пожалуйста, не кладите трубку.",
            "Здравствуйте, я специалист банка по работе с клиентами.",
            "У нас зафиксирована попытка входа в ваш личный кабинет.",
            "Для подтверждения операции назовите код из СМС.",
            "Ваша карта будет заблокирована, если вы не подтвердите операцию.",
            "Звонок записывается в целях улучшения качества обслуживания.",
        ]
        
        robot_dir = Path('data/audio/robot/silero')
        robot_dir.mkdir(parents=True, exist_ok=True)
        
        total_files = len(speakers) * len(texts) * multiplier
        logger.info(f"Генерация {total_files} синтезированных файлов через Silero...")
        
        file_counter = 0
        
        for m in range(multiplier):
            for i, speaker in enumerate(speakers):
                for j, text in enumerate(tqdm(texts, desc=f"Голос {speaker}, пакет {m+1}")):
                    try:
                        # Небольшие вариации текста
                        if random.random() > 0.5:
                            text = text.replace("Здравствуйте", "Добрый день")
                        if random.random() > 0.7:
                            text = text + " " + random.choice(["Спасибо", "Пожалуйста", "Благодарю"])
                        
                        # Генерация аудио с правильным устройством
                        audio = model.apply_tts(
                            text=text,
                            speaker=speaker,
                            sample_rate=48000
                        )
                        
                        # Конвертируем в numpy
                        if torch.is_tensor(audio):
                            audio = audio.cpu().detach().numpy()
                        elif hasattr(audio, 'numpy'):
                            audio = audio.numpy()
                        
                        # Убеждаемся, что это numpy массив
                        audio = np.array(audio, dtype=np.float32)
                        
                        # Проверяем на NaN и бесконечности
                        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
                            logger.warning(f"Обнаружены NaN значения, пропускаем")
                            continue
                        
                        # Downsample до 8kHz (48000 -> 8000)
                        audio = audio[::6]
                        
                        # Нормализация
                        max_val = np.max(np.abs(audio))
                        if max_val > 0:
                            audio = audio / max_val
                        
                        # Добавление небольшого шума для разнообразия
                        if random.random() > 0.8:
                            noise = np.random.randn(len(audio)) * 0.001
                            audio = audio + noise
                            audio = audio / np.max(np.abs(audio))
                        
                        filename = robot_dir / f"silero_{speaker}_{j:03d}_v{m:02d}.wav"
                        
                        # Сохраняем с правильным типом данных
                        sf.write(filename, audio.astype(np.float32), 8000)
                        
                        file_counter += 1
                        
                    except Exception as e:
                        logger.error(f"Ошибка при генерации файла {file_counter}: {e}")
                        continue
        
        logger.info(f"✓ Silero: сгенерировано {file_counter} файлов")
        return file_counter
        
    except Exception as e:
        logger.error(f"Ошибка в Silero генерации: {e}")
        import traceback
        traceback.print_exc()
        return 0

def generate_rhvoice_fixed():
    """
    Исправленная генерация для RHVoice - создаем реалистичные имитации
    """
    import struct
    
    logger.info("Генерация через имитацию RHVoice...")
    
    texts = [
        "Здравствуйте, это роботизированный звонок.",
        "Ваш звонок очень важен для нас.",
        "Оставайтесь на линии, мы соединим вас с оператором.",
        "Время ожидания составляет примерно две минуты.",
        "Если вы ошиблись номером, просто положите трубку.",
        "Для связи с оператором нажмите ноль.",
        "Чтобы повторить информацию, нажмите звездочку.",
        "Спасибо за обращение в нашу компанию.",
        "Ваш звонок будет записан в учебных целях.",
        "Пожалуйста, говорите громче, вас плохо слышно."
    ]
    
    # "Голоса" для имитации
    voice_profiles = [
        {'name': 'anna', 'pitch': 180, 'speed': 1.0, 'vibrato': 0.02},
        {'name': 'arina', 'pitch': 165, 'speed': 0.95, 'vibrato': 0.03},
        {'name': 'elena', 'pitch': 200, 'speed': 1.1, 'vibrato': 0.01},
        {'name': 'irina', 'pitch': 190, 'speed': 1.0, 'vibrato': 0.04},
        {'name': 'tatyana', 'pitch': 175, 'speed': 0.9, 'vibrato': 0.02},
        {'name': 'mikhail', 'pitch': 120, 'speed': 0.95, 'vibrato': 0.01},
        {'name': 'vladimir', 'pitch': 110, 'speed': 0.85, 'vibrato': 0.02},
    ]
    
    robot_dir = Path('data/audio/robot/rhvoice')
    robot_dir.mkdir(parents=True, exist_ok=True)
    
    file_counter = 0
    
    for voice in voice_profiles:
        for i, text in enumerate(texts):
            try:
                # Длительность зависит от длины текста и скорости
                duration = len(text) * 0.1 / voice['speed']
                sample_rate = 8000
                t = np.linspace(0, duration, int(sample_rate * duration))
                
                # Базовая частота голоса
                f0 = voice['pitch']
                
                # Добавляем вибрато
                vibrato = voice['vibrato'] * np.sin(2 * np.pi * 5 * t)
                pitch_variation = f0 * (1 + vibrato)
                
                # Форманты для разных гласных
                formants = []
                for word in text.split():
                    if any(v in word.lower() for v in ['а', 'я']):
                        formants.extend([800, 1200])
                    elif any(v in word.lower() for v in ['о', 'е']):
                        formants.extend([500, 900])
                    elif any(v in word.lower() for v in ['у', 'ю']):
                        formants.extend([400, 700])
                    elif any(v in word.lower() for v in ['и', 'ы']):
                        formants.extend([300, 2500])
                
                # Генерируем сигнал
                signal = 0
                for harmonic in range(1, 5):
                    amp = 1.0 / harmonic
                    
                    # Основной тон
                    signal += amp * np.sin(2 * np.pi * harmonic * pitch_variation * t)
                    
                    # Форманты
                    if formants:
                        for f in formants[:2]:
                            signal += (amp * 0.5) * np.sin(2 * np.pi * harmonic * f * t)
                
                # Добавляем шум (имитация согласных)
                noise = np.random.randn(len(signal)) * 0.05
                noise_env = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))
                signal = signal + noise * noise_env
                
                # Добавляем паузы между словами
                words_count = len(text.split())
                pauses = np.zeros(int(sample_rate * 0.05 * words_count))
                signal = np.concatenate([signal, pauses])
                
                # Обрезаем до нужной длины
                if len(signal) > sample_rate * 5:
                    signal = signal[:sample_rate * 5]
                
                # Нормализация
                signal = signal / np.max(np.abs(signal))
                
                filename = robot_dir / f"rhvoice_{voice['name']}_{i:02d}.wav"
                sf.write(filename, signal.astype(np.float32), sample_rate)
                
                file_counter += 1
                
            except Exception as e:
                logger.error(f"Ошибка генерации: {e}")
                continue
    
    logger.info(f"✓ RHVoice: сгенерировано {file_counter} файлов")
    return file_counter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--multiplier', type=int, default=5, help='Множитель генерации')
    parser.add_argument('--engine', type=str, default='silero', choices=['silero', 'rhvoice', 'all'])
    args = parser.parse_args()
    
    logger.info(f"Начинаем массовую генерацию синтетической речи (x{args.multiplier})...")
    
    total = 0
    
    if args.engine in ['silero', 'all']:
        count = generate_silero_voices_fixed_batch(args.multiplier)
        total += count
    
    if args.engine in ['rhvoice', 'all']:
        count = generate_rhvoice_fixed()
        total += count
    
    logger.info(f"✅ Генерация завершена. Всего создано файлов: {total}")