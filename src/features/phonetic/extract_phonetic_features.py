import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')


class PhoneticFeatureExtractor:
    def __init__(self, sample_rate=8000, n_fft=512, hop_length=256):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.feature_dim = 27
    
    def _find_peaks(self, spectrum, min_distance=3):
        """Поиск локальных максимумов спектра"""
        peaks = []
        for i in range(min_distance, len(spectrum) - min_distance):
            if (spectrum[i] > spectrum[i-1] and 
                spectrum[i] > spectrum[i+1] and
                spectrum[i] > spectrum[i-2] and
                spectrum[i] > spectrum[i+2]):
                peaks.append(i)
        
        if peaks:
            peaks_vals = [(p, spectrum[p]) for p in peaks]
            peaks_vals.sort(key=lambda x: x[1], reverse=True)
            return [p for p, _ in peaks_vals[:3]]
        return []
    
    def extract_formants(self, y, sr):
        """Извлечение формант (F1, F2, F3)"""
        D = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        formants = []
        
        for i in range(D.shape[1]):
            spectrum = D[:, i]
            peaks = self._find_peaks(spectrum)
            if len(peaks) >= 3:
                formants.append(peaks[:3])
            else:
                formants.append([0, 0, 0])
        
        formants = np.array(formants)
        
        features = []
        for i in range(3):
            features.append(np.mean(formants[:, i]))
            features.append(np.std(formants[:, i]))
        
        return np.array(features)
    
    def extract_vot(self, y, sr):
        """Voice Onset Time"""
        envelope = np.abs(y)
        window_size = int(sr * 0.01)
        if len(envelope) > window_size:
            envelope = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
        
        derivative = np.diff(envelope)
        threshold = np.std(derivative) * 2
        onsets = np.where(np.abs(derivative) > threshold)[0]
        
        if len(onsets) > 1:
            vot_values = np.diff(onsets) / sr
            features = [
                np.mean(vot_values),
                np.std(vot_values),
                np.max(vot_values),
                np.min(vot_values),
                len(vot_values)
            ]
        else:
            features = [0, 0, 0, 0, 0]
        
        return np.array(features)
    
    def extract_spectral_envelope(self, y, sr):
        """Спектральная огибающая"""
        spectrum = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), 1/sr)
        mask = freqs <= 4000
        spectrum = spectrum[mask]
        
        log_spectrum = np.log(spectrum + 1e-10)
        derivative = np.diff(log_spectrum)
        
        features = [
            np.mean(log_spectrum),
            np.std(log_spectrum),
            np.mean(np.abs(derivative)),
            np.std(derivative),
            np.max(derivative),
            np.min(derivative)
        ]
        
        return np.array(features)
    
    def extract_hnr(self, y, sr):
        """Harmonics-to-Noise Ratio"""
        harmonic = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)
        
        energy_harmonic = np.sum(harmonic ** 2)
        energy_noise = np.sum(percussive ** 2)
        energy_total = energy_harmonic + energy_noise
        
        hnr_overall = energy_harmonic / (energy_total + 1e-10)
        
        frame_length = int(sr * 0.025)
        hop_length = int(sr * 0.010)
        
        hnr_frames = []
        for start in range(0, len(y) - frame_length, hop_length):
            frame = y[start:start + frame_length]
            harm = librosa.effects.harmonic(frame)
            perc = librosa.effects.percussive(frame)
            
            e_harm = np.sum(harm ** 2)
            e_noise = np.sum(perc ** 2)
            e_total = e_harm + e_noise
            
            hnr_frames.append(e_harm / (e_total + 1e-10))
        
        features = [
            hnr_overall,
            np.mean(hnr_frames),
            np.std(hnr_frames),
            np.max(hnr_frames),
            np.min(hnr_frames)
        ]
        
        return np.array(features)
    
    def extract_all(self, audio_path):
        """Извлечение всех 27 признаков"""
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=5)
            if len(y) == 0:
                return None
            
            y = y / (np.max(np.abs(y)) + 1e-10)
            features = []
            
            # 1. Форманты (6)
            formants = self.extract_formants(y, sr)
            features.extend(formants)
            
            # 2. VOT (5)
            vot = self.extract_vot(y, sr)
            features.extend(vot)
            
            # 3. Спектральная огибающая (6)
            envelope = self.extract_spectral_envelope(y, sr)
            features.extend(envelope)
            
            # 4. HNR (5)
            hnr = self.extract_hnr(y, sr)
            features.extend(hnr)
            
            # 5. Формантные соотношения (5)
            if len(formants) >= 6:
                f1_mean = formants[0]
                f2_mean = formants[2]
                f3_mean = formants[4]
                features.append(f2_mean / (f1_mean + 1e-10))
                features.append(f3_mean / (f2_mean + 1e-10))
                features.append(f1_mean / (f2_mean + 1e-10))
                features.append(f2_mean - f1_mean)
                features.append(f3_mean - f2_mean)
            else:
                features.extend([0, 0, 0, 0, 0])
            
            # Проверка размерности
            if len(features) != self.feature_dim:
                if len(features) < self.feature_dim:
                    features.extend([0] * (self.feature_dim - len(features)))
                else:
                    features = features[:self.feature_dim]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Error extracting phonetic features: {e}")
            return None
    
    def extract_batch(self, audio_paths, output_path):
        """Последовательное извлечение"""
        features = []
        for path in tqdm(audio_paths, desc="Extracting phonetic"):
            feats = self.extract_all(path)
            if feats is not None:
                features.append(feats)
        
        features = np.array(features)
        np.save(output_path, features)
        return features
    
    def extract_batch_parallel(self, audio_paths, output_path, n_workers=None):
        """Параллельное извлечение признаков"""
        n_workers = n_workers or cpu_count()
        
        with Pool(processes=n_workers) as pool:
            features = list(tqdm(
                pool.imap(self.extract_all, audio_paths),
                total=len(audio_paths),
                desc="Extracting phonetic"
            ))
        
        features = [f for f in features if f is not None]
        
        features_array = np.array(features)
        np.save(output_path, features_array)
        print(f"  Saved {len(features_array)} phonetic features to {output_path}")
        return features_array