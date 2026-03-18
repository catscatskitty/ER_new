import librosa
import numpy as np
import parselmouth
from typing import List
from ..base_extractor import BaseExtractor

class ProsodicExtractor(BaseExtractor):
    """
    Извлечение просодических признаков (интонация, темп, ритм)
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Извлечение просодических признаков
        """
        features = []
        
        # Zero crossing rate (темп)
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.extend([
            np.mean(zcr),
            np.std(zcr),
            np.max(zcr),
            np.min(zcr)
        ])
        
        # RMS energy (громкость)
        rms = librosa.feature.rms(y=audio)
        features.extend([
            np.mean(rms),
            np.std(rms),
            np.max(rms),
            np.min(rms)
        ])
        
        # Pitch (F0) using parselmouth
        try:
            snd = parselmouth.Sound(audio, sampling_frequency=sr)
            pitch = snd.to_pitch()
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values != 0]  # Remove unvoiced
            
            if len(pitch_values) > 0:
                features.extend([
                    np.mean(pitch_values),
                    np.std(pitch_values),
                    np.max(pitch_values),
                    np.min(pitch_values),
                    np.percentile(pitch_values, 25),
                    np.percentile(pitch_values, 75)
                ])
            else:
                features.extend([0, 0, 0, 0, 0, 0])
        except:
            features.extend([0, 0, 0, 0, 0, 0])
        
        # Formants (using LPC)
        try:
            lpc = librosa.lpc(audio, order=8)
            roots = np.roots(lpc)
            roots = roots[np.imag(roots) >= 0]
            angles = np.arctan2(np.imag(roots), np.real(roots))
            formants = sorted(angles * (sr / (2 * np.pi)))
            
            for i, f in enumerate(formants[:4]):  # First 4 formants
                features.append(f)
            for i in range(4 - len(formants)):  # Pad if less than 4
                features.append(0)
        except:
            features.extend([0, 0, 0, 0])
        
        # Jitter (микроизменения частоты)
        try:
            jitter = self._extract_jitter(audio, sr)
            features.extend(jitter)
        except:
            features.extend([0, 0, 0])
        
        # Shimmer (микроизменения амплитуды)
        try:
            shimmer = self._extract_shimmer(audio, sr)
            features.extend(shimmer)
        except:
            features.extend([0, 0, 0])
        
        return np.array(features)
    
    def _extract_jitter(self, audio: np.ndarray, sr: int) -> List[float]:
        """
        Извлечение джиттера (вариации частоты)
        """
        # Упрощенная реализация
        f0, voiced_flag, _ = librosa.pyin(audio, fmin=75, fmax=300, sr=sr)
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) < 3:
            return [0, 0, 0]
        
        # Jitter metrics
        diffs = np.abs(np.diff(f0_clean))
        jitter_abs = np.mean(diffs)
        jitter_rel = jitter_abs / np.mean(f0_clean)
        jitter_rap = np.mean(np.abs(f0_clean[2:] - 2*f0_clean[1:-1] + f0_clean[:-2])) / np.mean(f0_clean)
        
        return [jitter_abs, jitter_rel, jitter_rap]
    
    def _extract_shimmer(self, audio: np.ndarray, sr: int) -> List[float]:
        """
        Извлечение шиммера (вариации амплитуды)
        """
        # Амплитудная огибающая
        rms = librosa.feature.rms(y=audio)[0]
        
        if len(rms) < 3:
            return [0, 0, 0]
        
        # Shimmer metrics
        diffs = np.abs(np.diff(rms))
        shimmer_abs = np.mean(diffs)
        shimmer_rel = shimmer_abs / np.mean(rms)
        shimmer_apq = np.std(rms) / np.mean(rms)
        
        return [shimmer_abs, shimmer_rel, shimmer_apq]
    
    def get_feature_names(self) -> List[str]:
        """
        Получение названий признаков
        """
        return [
            'zcr_mean', 'zcr_std', 'zcr_max', 'zcr_min',
            'rms_mean', 'rms_std', 'rms_max', 'rms_min',
            'pitch_mean', 'pitch_std', 'pitch_max', 'pitch_min',
            'pitch_q25', 'pitch_q75',
            'formant1', 'formant2', 'formant3', 'formant4',
            'jitter_abs', 'jitter_rel', 'jitter_rap',
            'shimmer_abs', 'shimmer_rel', 'shimmer_apq'
        ]