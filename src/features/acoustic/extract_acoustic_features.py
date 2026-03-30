import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')


class AcousticFeatureExtractor:
    def __init__(self, sr=8000, n_mfcc=13, n_fft=512, hop_length=256):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.feature_dim = 38
    
    def extract_from_file(self, audio_path):
        """Извлечение признаков из одного файла"""
        try:
            y, sr = librosa.load(audio_path, sr=self.sr, duration=5)
            if len(y) == 0:
                return None
            
            y = y / (np.max(np.abs(y)) + 1e-10)
            features = []
            
            # 1. MFCC (26)
            mfcc = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=self.n_mfcc,
                n_fft=self.n_fft, hop_length=self.hop_length
            )
            features.extend(np.mean(mfcc, axis=1))
            features.extend(np.std(mfcc, axis=1))
            
            # 2. Спектральные признаки (3)
            try:
                centroid = np.mean(librosa.feature.spectral_centroid(
                    y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
                ))
                features.append(centroid)
            except:
                features.append(0)
            
            try:
                bandwidth = np.mean(librosa.feature.spectral_bandwidth(
                    y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
                ))
                features.append(bandwidth)
            except:
                features.append(0)
            
            try:
                rolloff = np.mean(librosa.feature.spectral_rolloff(
                    y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
                ))
                features.append(rolloff)
            except:
                features.append(0)
            
            # 3. Просодические признаки (3)
            try:
                zcr = np.mean(librosa.feature.zero_crossing_rate(
                    y, frame_length=self.n_fft, hop_length=self.hop_length
                ))
                features.append(zcr)
            except:
                features.append(0)
            
            try:
                rms = np.mean(librosa.feature.rms(
                    y=y, frame_length=self.n_fft, hop_length=self.hop_length
                ))
                features.append(rms)
            except:
                features.append(0)
            
            try:
                tempo = librosa.beat.tempo(y=y, sr=sr, hop_length=self.hop_length)[0]
                features.append(float(tempo))
            except:
                features.append(0)
            
            # 4. Chroma (6)
            try:
                chroma = librosa.feature.chroma_stft(
                    y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
                )
                chroma_mean = np.mean(chroma, axis=1)
                features.extend(chroma_mean[:6])
            except:
                features.extend([0, 0, 0, 0, 0, 0])
            
            # Проверка размерности
            if len(features) != self.feature_dim:
                if len(features) < self.feature_dim:
                    features.extend([0] * (self.feature_dim - len(features)))
                else:
                    features = features[:self.feature_dim]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Error extracting from {audio_path}: {e}")
            return None
    
    def extract_batch(self, audio_paths, output_path):
        """Последовательное извлечение признаков"""
        features = []
        for path in tqdm(audio_paths, desc="Extracting acoustic"):
            feats = self.extract_from_file(path)
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
                pool.imap(self.extract_from_file, audio_paths),
                total=len(audio_paths),
                desc="Extracting acoustic"
            ))
        
        # Фильтруем None
        features = [f for f in features if f is not None]
        
        features_array = np.array(features)
        np.save(output_path, features_array)
        print(f"  Saved {len(features_array)} acoustic features to {output_path}")
        return features_array