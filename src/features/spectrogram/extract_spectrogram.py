import numpy as np
import librosa
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


class SpectrogramExtractor:
    def __init__(self, sr=8000, n_mels=128, n_fft=512, hop_length=256, duration=5):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
    
    def extract_from_file(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)
            if len(y) == 0:
                return None
            
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=self.n_mels,
                n_fft=self.n_fft, hop_length=self.hop_length
            )
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Нормализация
            log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
            
            return log_mel.astype(np.float32)
            
        except Exception as e:
            print(f"Error extracting spectrogram: {e}")
            return None
    
    def extract_batch(self, audio_paths, output_path, pad_to_max=True):
        """Последовательное извлечение"""
        specs = []
        lengths = []
        
        for path in tqdm(audio_paths, desc="Extracting spectrograms"):
            spec = self.extract_from_file(path)
            if spec is not None:
                specs.append(spec)
                lengths.append(spec.shape[1])
        
        if pad_to_max and specs:
            max_len = max(lengths)
            padded = []
            for spec in specs:
                if spec.shape[1] < max_len:
                    pad_width = max_len - spec.shape[1]
                    spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
                padded.append(spec)
            specs = padded
        
        specs_array = np.array(specs)
        np.save(output_path, specs_array)
        return specs_array
    
    def extract_batch_parallel(self, audio_paths, output_path, pad_to_max=True, n_workers=None):
        """Параллельное извлечение"""
        n_workers = n_workers or cpu_count()
        
        with Pool(processes=n_workers) as pool:
            specs = list(tqdm(
                pool.imap(self.extract_from_file, audio_paths),
                total=len(audio_paths),
                desc="Extracting spectrograms"
            ))
        
        specs = [s for s in specs if s is not None]
        
        if pad_to_max and specs:
            max_len = max(s.shape[1] for s in specs)
            padded = []
            for spec in specs:
                if spec.shape[1] < max_len:
                    pad_width = max_len - spec.shape[1]
                    spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
                padded.append(spec)
            specs = padded
        
        specs_array = np.array(specs)
        np.save(output_path, specs_array)
        print(f"  Saved {len(specs_array)} spectrograms to {output_path}")
        return specs_array