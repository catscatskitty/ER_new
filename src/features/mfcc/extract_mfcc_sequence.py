import numpy as np
import librosa
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


class MFCCSequenceExtractor:
    def __init__(self, sr=8000, n_mfcc=13, n_fft=512, hop_length=256, duration=5):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
    
    def extract_from_file(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)
            if len(y) == 0:
                return None
            
            mfcc = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=self.n_mfcc,
                n_fft=self.n_fft, hop_length=self.hop_length
            )
            mfcc = mfcc.T  # (time, n_mfcc)
            
            # Нормализация
            mfcc = (mfcc - mfcc.mean(axis=0)) / (mfcc.std(axis=0) + 1e-8)
            
            return mfcc.astype(np.float32)
            
        except Exception as e:
            print(f"Error extracting MFCC: {e}")
            return None
    
    def extract_batch(self, audio_paths, output_path, pad_to_max=True):
        """Последовательное извлечение"""
        sequences = []
        lengths = []
        
        for path in tqdm(audio_paths, desc="Extracting MFCC sequences"):
            seq = self.extract_from_file(path)
            if seq is not None:
                sequences.append(seq)
                lengths.append(seq.shape[0])
        
        if pad_to_max and sequences:
            max_len = max(lengths)
            padded = []
            for seq in sequences:
                if seq.shape[0] < max_len:
                    pad_width = max_len - seq.shape[0]
                    seq = np.pad(seq, ((0, pad_width), (0, 0)), mode='constant')
                padded.append(seq)
            sequences = padded
        
        seqs_array = np.array(sequences)
        np.save(output_path, seqs_array)
        return seqs_array
    
    def extract_batch_parallel(self, audio_paths, output_path, pad_to_max=True, n_workers=None):
        """Параллельное извлечение"""
        n_workers = n_workers or cpu_count()
        
        with Pool(processes=n_workers) as pool:
            sequences = list(tqdm(
                pool.imap(self.extract_from_file, audio_paths),
                total=len(audio_paths),
                desc="Extracting MFCC sequences"
            ))
        
        sequences = [s for s in sequences if s is not None]
        
        if pad_to_max and sequences:
            max_len = max(s.shape[0] for s in sequences)
            padded = []
            for seq in sequences:
                if seq.shape[0] < max_len:
                    pad_width = max_len - seq.shape[0]
                    seq = np.pad(seq, ((0, pad_width), (0, 0)), mode='constant')
                padded.append(seq)
            sequences = padded
        
        seqs_array = np.array(sequences)
        np.save(output_path, seqs_array)
        print(f"  Saved {len(seqs_array)} MFCC sequences to {output_path}")
        return seqs_array