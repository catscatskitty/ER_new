import librosa
import numpy as np
from ..base_extractor import BaseExtractor

class MFCCExtractor(BaseExtractor):
    """
    Извлечение MFCC признаков
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.n_mfcc = config['acoustic']['mfcc']['n_mfcc']
        self.n_fft = config['acoustic']['mfcc']['n_fft']
        self.hop_length = config['acoustic']['mfcc']['hop_length']
        self.n_mels = config['acoustic']['mfcc']['n_mels']
        
    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Извлечение MFCC признаков
        """
        # MFCC
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Статистики по времени
        features = []
        features.extend(np.mean(mfccs, axis=1))  # средние
        features.extend(np.std(mfccs, axis=1))   # стандартные отклонения
        
        # Дельта и дельта-дельта коэффициенты
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        features.extend(np.mean(mfcc_delta, axis=1))
        features.extend(np.std(mfcc_delta, axis=1))
        features.extend(np.mean(mfcc_delta2, axis=1))
        features.extend(np.std(mfcc_delta2, axis=1))
        
        return np.array(features)
    
    def get_feature_names(self) -> list:
        """
        Получение названий признаков
        """
        names = []
        for i in range(self.n_mfcc):
            names.append(f'mfcc_mean_{i+1}')
        for i in range(self.n_mfcc):
            names.append(f'mfcc_std_{i+1}')
        for i in range(self.n_mfcc):
            names.append(f'mfcc_delta_mean_{i+1}')
        for i in range(self.n_mfcc):
            names.append(f'mfcc_delta_std_{i+1}')
        for i in range(self.n_mfcc):
            names.append(f'mfcc_delta2_mean_{i+1}')
        for i in range(self.n_mfcc):
            names.append(f'mfcc_delta2_std_{i+1}')
        return names