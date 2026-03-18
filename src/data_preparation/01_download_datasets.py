import os
import requests
import zipfile
import tarfile
from pathlib import Path
import logging
from tqdm import tqdm
import shutil
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url: str, dest_path: Path, max_retries: int = 3):
    """
    Скачивание файла с прогресс-баром и повторными попытками
    """
    for attempt in range(max_retries):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, stream=True, timeout=30, headers=headers, allow_redirects=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
                    for data in response.iter_content(chunk_size=8192):
                        f.write(data)
                        pbar.update(len(data))
            return True
            
        except Exception as e:
            logger.warning(f"Попытка {attempt + 1}/{max_retries} не удалась для {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            continue
    
    logger.error(f"Не удалось скачать {url} после {max_retries} попыток")
    return False

def download_from_hf(repo_id: str, filename: str, dest_path: Path):
    """
    Скачивание файла с Hugging Face
    """
    try:
        from huggingface_hub import hf_hub_download
        
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset"
        )
        
        # Копируем в нужную директорию
        shutil.copy2(downloaded_path, dest_path)
        return True
        
    except ImportError:
        logger.error("Установите huggingface-hub: pip install huggingface-hub")
        return False
    except Exception as e:
        logger.error(f"Ошибка скачивания с Hugging Face: {e}")
        return False

def download_ruslun():
    """
    Скачивание датасета ruSLUn с Hugging Face
    """
    logger.info("Скачивание датасета ruSLUn (реальные записи русской речи)...")
    
    try:
        # Пытаемся установить huggingface-hub если нет
        import subprocess
        import sys
        
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            logger.info("Установка huggingface-hub...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface-hub"])
            from huggingface_hub import snapshot_download
        
        human_dir = Path('data/audio/human')
        human_dir.mkdir(parents=True, exist_ok=True)
        
        # Скачиваем датасет
        logger.info("Загрузка датасета (это может занять некоторое время)...")
        
        # Скачиваем только wav файлы для экономии времени
        snapshot_download(
            repo_id="MERA-evaluation/ruSLUn",
            local_dir=human_dir / "ruslun_temp",
            allow_patterns=["*.wav"],
            max_workers=4
        )
        
        # Перемещаем wav файлы в основную директорию
        wav_files = list((human_dir / "ruslun_temp").rglob("*.wav"))
        for wav_file in wav_files:
            shutil.copy2(wav_file, human_dir / wav_file.name)
        
        # Удаляем временную папку
        shutil.rmtree(human_dir / "ruslun_temp")
        
        logger.info(f"✅ Скачано {len(wav_files)} файлов из ruSLUn")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка скачивания ruSLUn: {e}")
        return False

def download_common_voice_ru():
    """
    Скачивание русской части Common Voice
    """
    logger.info("Скачивание Common Voice Russian...")
    
    # Прямые ссылки на русскую часть Common Voice
    urls = [
        "https://mozilla-common-voice-datasets.s3.amazonaws.com/cv-corpus-16.1-2023-12-06/cv-corpus-16.1-2023-12-06-ru.tar.gz",
        "https://storage.googleapis.com/common-voice-prod-prod-datasets/cv-corpus-15.0-2023-09-08/cv-corpus-15.0-2023-09-08-ru.tar.gz"
    ]
    
    human_dir = Path('data/audio/human')
    human_dir.mkdir(parents=True, exist_ok=True)
    
    for url in urls:
        try:
            filename = url.split('/')[-1]
            temp_file = human_dir / filename
            
            if download_file(url, temp_file):
                # Распаковка
                import tarfile
                with tarfile.open(temp_file, 'r:gz') as tar:
                    tar.extractall(human_dir / "common_voice_temp")
                
                # Копируем аудио
                audio_files = list((human_dir / "common_voice_temp").rglob("*.mp3"))
                audio_files.extend((human_dir / "common_voice_temp").rglob("*.wav"))
                
                for audio_file in audio_files[:50]:  # Берем первые 50 для теста
                    shutil.copy2(audio_file, human_dir / audio_file.name)
                
                # Очистка
                shutil.rmtree(human_dir / "common_voice_temp")
                temp_file.unlink()
                
                logger.info(f"✅ Скачано {len(audio_files[:50])} файлов из Common Voice")
                return True
                
        except Exception as e:
            logger.warning(f"Не удалось скачать Common Voice: {e}")
            continue
    
    return False

def download_golos():
    """
    Скачивание датасета Golos (русская речь от Сбера)
    """
    logger.info("Скачивание Golos dataset...")
    
    # Hugging Face версия Golos
    try:
        from huggingface_hub import snapshot_download
        
        human_dir = Path('data/audio/human')
        human_dir.mkdir(parents=True, exist_ok=True)
        
        snapshot_download(
            repo_id="bond005/sberdevices_golos_100h_crowd",
            local_dir=human_dir / "golos_temp",
            allow_patterns=["*.wav"],
            max_workers=4
        )
        
        wav_files = list((human_dir / "golos_temp").rglob("*.wav"))
        for wav_file in wav_files[:100]:  # Берем первые 100
            shutil.copy2(wav_file, human_dir / wav_file.name)
        
        shutil.rmtree(human_dir / "golos_temp")
        logger.info(f"✅ Скачано {min(100, len(wav_files))} файлов из Golos")
        return True
        
    except Exception as e:
        logger.warning(f"Не удалось скачать Golos: {e}")
        return False

def download_ruslan():
    """
    Скачивание RUSLAN датасета
    """
    logger.info("Скачивание RUSLAN dataset...")
    
    # Альтернативный источник RUSLAN
    url = "https://huggingface.co/datasets/Den4ikAI/rustts/resolve/main/ruslan.zip"
    
    human_dir = Path('data/audio/human')
    human_dir.mkdir(parents=True, exist_ok=True)
    
    temp_file = human_dir / "ruslan.zip"
    
    if download_file(url, temp_file):
        try:
            with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                zip_ref.extractall(human_dir / "ruslan_temp")
            
            wav_files = list((human_dir / "ruslan_temp").rglob("*.wav"))
            for wav_file in wav_files[:50]:  # Берем первые 50
                shutil.copy2(wav_file, human_dir / wav_file.name)
            
            shutil.rmtree(human_dir / "ruslan_temp")
            temp_file.unlink()
            
            logger.info(f"✅ Скачано {min(50, len(wav_files))} файлов из RUSLAN")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка распаковки RUSLAN: {e}")
            return False
    
    return False

def download_mls_russian():
    """
    Скачивание MLS Russian (Multilingual LibriSpeech)
    """
    logger.info("Скачивание MLS Russian...")
    
    # Маленькая тестовая часть MLS
    url = "https://dl.fbaipublicfiles.com/mls/mls_russian_opus.tar.gz"
    
    human_dir = Path('data/audio/human')
    human_dir.mkdir(parents=True, exist_ok=True)
    
    temp_file = human_dir / "mls_russian.tar.gz"
    
    if download_file(url, temp_file):
        try:
            import tarfile
            with tarfile.open(temp_file, 'r:gz') as tar:
                tar.extractall(human_dir / "mls_temp")
            
            # Берем только тестовую часть для экономии места
            audio_files = list((human_dir / "mls_temp/mls_russian/test").rglob("*.opus"))
            
            # Конвертируем opus в wav (если есть ffmpeg)
            converted = 0
            for audio_file in audio_files[:20]:
                wav_file = human_dir / f"mls_{audio_file.stem}.wav"
                try:
                    import subprocess
                    subprocess.run([
                        'ffmpeg', '-i', str(audio_file), 
                        '-acodec', 'pcm_s16le', 
                        '-ar', '8000', 
                        '-ac', '1', 
                        str(wav_file)
                    ], check=True, capture_output=True)
                    converted += 1
                except:
                    # Если ffmpeg нет, пропускаем
                    pass
            
            shutil.rmtree(human_dir / "mls_temp")
            temp_file.unlink()
            
            logger.info(f"✅ Скачано и сконвертировано {converted} файлов из MLS")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обработки MLS: {e}")
            return False
    
    return False

def download_open_datasets():
    """
    Скачивание всех доступных датасетов
    """
    logger.info("="*60)
    logger.info("НАЧИНАЕМ СКАЧИВАНИЕ ДАТАСЕТОВ")
    logger.info("="*60)
    
    human_dir = Path('data/audio/human')
    human_dir.mkdir(parents=True, exist_ok=True)
    
    total_downloaded = 0
    
    # 1. ruSLUn (лучший вариант)
    if download_ruslun():
        total_downloaded += 1
    
    # 2. Common Voice Russian
    if download_common_voice_ru():
        total_downloaded += 1
    
    # 3. Golos
    if download_golos():
        total_downloaded += 1
    
    # 4. RUSLAN
    if download_ruslan():
        total_downloaded += 1
    
    # 5. MLS Russian
    if download_mls_russian():
        total_downloaded += 1
    
    # Подсчет итогов
    human_files = list(human_dir.glob('*.wav'))
    logger.info("="*60)
    logger.info(f"ИТОГИ ЗАГРУЗКИ:")
    logger.info(f"  Успешно загружено датасетов: {total_downloaded}")
    logger.info(f"  Всего файлов человеческой речи: {len(human_files)}")
    
    if len(human_files) == 0:
        logger.warning("Не удалось скачать ни одного файла. Создаем демо-файлы...")
        create_sample_human_files()
    else:
        logger.info("✅ Подготовка данных завершена успешно!")
    
    return human_files

def create_sample_human_files():
    """
    Создание демо-файлов человеческой речи (резервный вариант)
    """
    logger.info("Создание демо-файлов человеческой речи...")
    
    human_dir = Path('data/audio/human')
    human_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        import numpy as np
        import soundfile as sf
        
        texts = [
            "Здравствуйте, меня зовут Александр. Чем я могу вам помочь?",
            "Добрый день! Это компания Эр-Телеком. У нас для вас специальное предложение.",
            "Алло! Да, я слушаю. Что вы хотели узнать?",
            "Извините, я сейчас занят. Перезвоните пожалуйста позже.",
            "Спасибо за звонок! Всего доброго, до свидания."
        ]
        
        for i, text in enumerate(texts):
            duration = len(text) / 10
            t = np.linspace(0, duration, int(8000 * duration))
            
            # Более реалистичная имитация речи
            f0 = 120 + 20 * np.sin(2 * np.pi * 3 * t)  # Основной тон
            f1 = 500 + 100 * np.sin(2 * np.pi * 2 * t)  # Первая форманта
            f2 = 1500 + 200 * np.sin(2 * np.pi * 1.5 * t)  # Вторая форманта
            
            signal = 0
            for harmonic in range(1, 5):
                signal += (1/harmonic) * np.sin(2 * np.pi * harmonic * f0 * t)
                signal += (0.5/harmonic) * np.sin(2 * np.pi * harmonic * f1 * t)
                signal += (0.3/harmonic) * np.sin(2 * np.pi * harmonic * f2 * t)
            
            # Добавляем шум и паузы
            noise = np.random.randn(len(signal)) * 0.02
            signal = signal + noise
            
            # Нормализация
            signal = signal / np.max(np.abs(signal))
            
            filename = human_dir / f"demo_human_{i:02d}.wav"
            sf.write(filename, signal, 8000)
            logger.info(f"  Создан демо-файл: {filename}")
            
    except Exception as e:
        logger.error(f"Ошибка создания демо-файлов: {e}")

if __name__ == "__main__":
    # Установка необходимых библиотек
    import subprocess
    import sys
    
    required_packages = ['huggingface-hub', 'requests', 'tqdm']
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            logger.info(f"Установка {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Запуск загрузки
    download_open_datasets()