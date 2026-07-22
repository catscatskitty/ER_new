import asyncio
import edge_tts
import os
import random
import requests
from tqdm import tqdm
import soundfile as sf
import librosa

# List of synthetic voices to use
VOICES = [
    "ru-RU-DmitryNeural",
    "ru-RU-SvetlanaNeural",
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-GB-SoniaNeural"
]

SAMPLE_TEXTS_RU = [
    "Привет, как твои дела? Сегодня отличная погода для прогулки.",
    "Я искусственный интеллект, созданный для помощи людям в их повседневных задачах.",
    "Распознавание речи — это сложная задача, требующая больших вычислительных мощностей.",
    "Вчера я ходил в магазин и купил много свежих фруктов и овощей.",
    "Завтра будет важная встреча, на которой мы обсудим планы на будущее."
]

SAMPLE_TEXTS_EN = [
    "Hello there! How are you doing today? The weather is quite nice.",
    "I am an artificial intelligence designed to assist humans with their tasks.",
    "Speech recognition is a challenging field that requires significant compute.",
    "I went to the store yesterday and bought some fresh fruits and vegetables.",
    "We have an important meeting tomorrow to discuss our future plans."
]

HUMAN_DATA_URLS = [
    # Using some public domain samples for demonstration
    "https://www.kozco.com/tech/LRMonoPhase4.wav",
    "https://www.kozco.com/tech/audio/Sine-1000Hz-100ms.wav", # Not speech, but I'll find better ones
]

# Better source for human speech: LJSpeech samples
LJ_SAMPLES = [
    f"https://data.keithito.com/data/speech/LJSpeech-1.1/wavs/LJ001-000{i}.wav" for i in range(1, 10)
]

async def generate_robot_audio(output_dir, num_samples=20):
    """Disabled: Use provided verified samples."""
    pass

def download_human_audio(output_dir, num_samples=10):
    """Disabled: Use provided verified samples."""
    pass

if __name__ == "__main__":
    print("Data generation is disabled. Please place your audio files in data/raw/human and data/raw/robot.")
