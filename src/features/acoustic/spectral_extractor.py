import librosa
import numpy as np
from typing import List
from ..base_extractor import BaseExtractor

class SpectralExtractor(BaseExtractor):
    """
    Извлечение спектральных признаков
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.n_fft = config['acoustic']['mfcc']['n_fft']
        self.hop_length = config['acoustic']['mfcc']['hop_length']
        
    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Извлечение спектральных признаков
        """
        features = []
        
        # Spectral centroid
        cent = librosa.feature.spectral_centroid(
            y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        features.extend([
            np.mean(cent),
            np.std(cent),
            np.max(cent),
            np.min(cent)
        ])
        
        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        features.extend([
            np.mean(bandwidth),
            np.std(bandwidth),
            np.max(bandwidth),
            np.min(bandwidth)
        ])
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        features.extend([
            np.mean(rolloff),
            np.std(rolloff),
            np.max(rolloff),
            np.min(rolloff)
        ])
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(
            y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        features.extend([
            np.mean(contrast, axis=1).tolist(),
            np.std(contrast, axis=1).tolist()
        ])
        
        # Flatness
        flatness = librosa.feature.spectral_flatness(
            y=audio, n_fft=self.n_fft, hop_length=self.hop_length
        )
        features.extend([
            np.mean(flatness),
            np.std(flatness)
        ])
        
        return np.array([item for sublist in features for item in (sublist if isinstance(sublist, list) else [sublist])])
    
    def get_feature_names(self) -> List[str]:
        """
        Получение названий признаков
        """
        names = [
            'centroid_mean', 'centroid_std', 'centroid_max', 'centroid_min',
            'bandwidth_mean', 'bandwidth_std', 'bandwidth_max', 'bandwidth_min',
            'rolloff_mean', 'rolloff_std', 'rolloff_max', 'rolloff_min'
        ]
        
        for i in range(7):  # 7 bands for spectral contrast
            names.append(f'contrast_mean_{i}')
        for i in range(7):
            names.append(f'contrast_std_{i}')
            
        names.extend(['flatness_mean', 'flatness_std'])
        
        return names