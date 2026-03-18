import os
import torch
import soundfile as sf
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_silero_voices():
    """
    Генерация синтезированной речи с помощью Silero TTS
    """
    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Загрузка модели Silero
        model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                             model='silero_tts',
                                             language='ru',
                                             speaker='v4_ru')
        model.to(device)
        
        # Доступные голоса
        speakers = ['xenia', 'baya', 'kseniya', 'aidar', 'eugene', 'random']
        
        # Тексты для синтеза
        texts = [
            "Здравствуйте, это звонок из банка. Вас беспокоит служба безопасности.",
            "На вашей карте замечена подозрительная активность. Подтвердите операцию.",
            "Для продолжения разговора нажмите цифру один на клавиатуре телефона.",
            "Ваш номер был выбран для участия в акции. Подробности можно узнать у оператора.",
            "Спасибо за ожидание. Соединяю с оператором, пожалуйста, не кладите трубку."
        ]
        
        robot_dir = Path('data/audio/robot/silero')
        robot_dir.mkdir(parents=True, exist_ok=True)
        
        for i, speaker in enumerate(speakers[:3]):  # Используем первые 3 голоса
            logger.info(f"Генерация голоса {speaker}...")
            
            for j, text in enumerate(texts):
                audio = model.apply_tts(text=text,
                                       speaker=speaker,
                                       sample_rate=48000)
                
                # Конвертируем в numpy и уменьшаем частоту до 8кГц
                audio = audio.cpu().numpy()
                audio = audio[::6]  # Простой downsample с 48к до 8к
                
                # Нормализация
                audio = audio / np.max(np.abs(audio))
                
                filename = robot_dir / f"silero_{speaker}_{j:02d}.wav"
                sf.write(filename, audio, 8000)
                
        logger.info(f"✓ Silero: сгенерировано {len(speakers[:3]) * len(texts)} файлов")
        
    except Exception as e:
        logger.error(f"Ошибка при генерации Silero: {e}")

def generate_yandex_speechkit():
    """
    Генерация с помощью Yandex SpeechKit (API)
    """
    # Здесь будет код для Yandex SpeechKit API
    # Требуется API ключ
    pass

def generate_rhvoice():
    """
    Генерация с помощью RHVoice
    """
    try:
        import subprocess
        
        texts = [
            "Здравствуйте, вас беспокоит компания Эр-Телеком.",
            "У нас для вас специальное предложение по подключению интернета.",
            "Желаете ли вы подключить наше телевидение со скидкой?",
            "Нажмите кнопку два, чтобы повторить информацию.",
            "Дождитесь ответа оператора, пожалуйста."
        ]
        
        voices = ['anna', 'arina', 'elena', 'irina', 'tatyana']
        
        robot_dir = Path('data/audio/robot/rhvoice')
        robot_dir.mkdir(parents=True, exist_ok=True)
        
        for voice in voices[:3]:
            for i, text in enumerate(texts):
                output_file = robot_dir / f"rhvoice_{voice}_{i:02d}.wav"
                # Здесь команда для RHVoice через командную строку
                # subprocess.run(['RHVoice-test', '-p', voice, '-i', text, '-o', output_file])
                
        logger.info(f"✓ RHVoice: сгенерировано {len(voices[:3]) * len(texts)} файлов")
        
    except Exception as e:
        logger.error(f"Ошибка при генерации RHVoice: {e}")

if __name__ == "__main__":
    logger.info("Начинаем генерацию синтезированной речи...")
    
    generate_silero_voices()
    generate_rhvoice()
    # generate_yandex_speechkit()  # Требуется API ключ
    
    logger.info("Генерация завершена")