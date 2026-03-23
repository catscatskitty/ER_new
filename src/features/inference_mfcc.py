"""
Извлечение MFCC-последовательности из аудио для инференса.
Совместимо с обучающим скриптом `extract_mfcc_sequences.py`.
"""
import numpy as np
import librosa

def extract_mfcc_sequence(audio_path, sample_rate=8000, n_mfcc=13, hop_length=512, duration=5, fixed_time_steps=128):
    """
    Возвращает MFCC-последовательность (fixed_time_steps, n_mfcc).
    """
    try:
        y, sr = librosa.load(audio_path, sr=sample_rate, duration=duration)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        mfcc = mfcc.T  # (time, n_mfcc)
        time_len = mfcc.shape[0]
        if time_len > fixed_time_steps:
            mfcc = mfcc[:fixed_time_steps, :]
        elif time_len < fixed_time_steps:
            pad = fixed_time_steps - time_len
            mfcc = np.pad(mfcc, ((0, pad), (0,0)), mode='constant', constant_values=0)
        return mfcc
    except Exception as e:
        print(f"Ошибка в extract_mfcc_sequence: {e}")
        return None