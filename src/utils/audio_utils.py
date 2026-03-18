# src/utils/audio_utils.py
import numpy as np
import librosa
from typing import Tuple, Optional

def load_audio(file_path: str, target_sr: int = 8000) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Загрузка аудиофайла с ресемплингом до target_sr
    """
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return audio, sr
    except Exception as e:
        print(f"Ошибка загрузки {file_path}: {e}")
        return None, None

def extract_features(audio: np.ndarray, sr: int, config: dict) -> np.ndarray:
    """
    Извлечение всех акустических признаков
    """
    features = []
    
    # MFCC
    mfccs = librosa.feature.mfcc(
        y=audio, 
        sr=sr, 
        n_mfcc=config['acoustic']['mfcc']['n_mfcc'],
        n_fft=config['acoustic']['mfcc']['n_fft'],
        hop_length=config['acoustic']['mfcc']['hop_length']
    )
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))
    
    # Spectral centroid
    cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features.append(np.mean(cent))
    features.append(np.std(cent))
    
    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    features.append(np.mean(bandwidth))
    features.append(np.std(bandwidth))
    
    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))
    
    # RMS energy
    rms = librosa.feature.rms(y=audio)
    features.append(np.mean(rms))
    features.append(np.std(rms))
    
    # Pitch (F0)
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, 
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        f0_clean = f0[~np.isnan(f0)]
        if len(f0_clean) > 0:
            features.append(np.mean(f0_clean))
            features.append(np.std(f0_clean))
            features.append(len(f0_clean) / len(f0))  # voicing ratio
        else:
            features.extend([0, 0, 0])
    except:
        features.extend([0, 0, 0])
    
    return np.array(features)

def pad_or_truncate(audio: np.ndarray, target_length: int) -> np.ndarray:
    """
    Обрезка или дополнение аудио до target_length
    """
    if len(audio) > target_length:
        return audio[:target_length]
    elif len(audio) < target_length:
        padding = target_length - len(audio)
        return np.pad(audio, (0, padding), 'constant')
    return audio

def add_noise(audio: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
    """
    Добавление шума
    """
    noise = np.random.randn(len(audio)) * noise_level
    return audio + noise

def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """
    Изменение темпа
    """
    return librosa.effects.time_stretch(audio, rate=rate)

def change_volume(audio: np.ndarray, factor: float) -> np.ndarray:
    """
    Изменение громкости
    """
    return audio * factor