"""
Извлечение спектрограммы из аудио для инференса.
Совместимо с обучающим скриптом `extract_spectrograms.py`.
"""
import numpy as np
import librosa

def extract_spectrogram(audio_path, sample_rate=8000, n_mels=128, hop_length=512, duration=5, fixed_time_steps=128):
    """
    Возвращает спектрограмму (n_mels, fixed_time_steps) в виде numpy array.
    """
    try:
        y, sr = librosa.load(audio_path, sr=sample_rate, duration=duration)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        # Нормализация
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)
        # Фиксированная длина
        time_len = log_mel.shape[1]
        if time_len > fixed_time_steps:
            log_mel = log_mel[:, :fixed_time_steps]
        elif time_len < fixed_time_steps:
            pad = fixed_time_steps - time_len
            log_mel = np.pad(log_mel, ((0,0), (0,pad)), mode='constant', constant_values=0)
        return log_mel
    except Exception as e:
        print(f"Ошибка в extract_spectrogram: {e}")
        return None