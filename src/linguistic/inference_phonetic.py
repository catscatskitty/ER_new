"""
Извлечение фонетических признаков для инференса.
Использует PhoneticFeatureExtractor из src/features/phonetic/phonetic_features.py.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features.phonetic.phonetic_features import PhoneticFeatureExtractor

_extractor = None

def get_extractor(sample_rate=8000):
    global _extractor
    if _extractor is None:
        _extractor = PhoneticFeatureExtractor(sample_rate=sample_rate)
    return _extractor

def extract_phonetic_from_audio(audio_path, sample_rate=8000):
    try:
        extractor = get_extractor(sample_rate)
        features = extractor.extract_all(audio_path)
        if features is None:
            return np.zeros(27, dtype=np.float32)
        return features
    except Exception as e:
        print(f"Ошибка в extract_phonetic_from_audio: {e}")
        return np.zeros(27, dtype=np.float32)