#!/usr/bin/env python
"""
Аугментация аудио для 19 128 исходных файлов
Настройки оптимизированы для баланса качества и объёма
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import random
import warnings
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')


class AudioAugmentor:
    def __init__(self, input_dir='data/audio', output_dir='data/processed/augmented_8khz',
                 snr_levels=None, noise_types=None, augment_ratio=0.5, 
                 include_phone=True, n_workers=None):
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Настройки аугментации (оптимизированные для 19k файлов)
        self.snr_levels = snr_levels or [10, 20]      # 2 уровня SNR
        self.noise_types = noise_types or ['white', 'pink']  # 2 типа шума
        self.augment_ratio = augment_ratio             # 0.5 = 50% файлов
        self.include_phone = include_phone
        
        self.n_workers = n_workers or cpu_count()
        
        print("="*60)
        print("Audio Augmentation Configuration")
        print("="*60)
        print(f"Input files: {len(list(self.input_dir.rglob('*.wav')))}")
        print(f"SNR levels: {self.snr_levels}")
        print(f"Noise types: {self.noise_types}")
        print(f"Augment ratio: {self.augment_ratio * 100}%")
        print(f"Phone filter: {self.include_phone}")
        print(f"Workers: {self.n_workers}")
        print("="*60)
        
        # Инициализация шумов
        self._init_noises()
    
    def _init_noises(self):
        """Инициализация функций шума"""
        self.noise_funcs = {}
        
        if 'white' in self.noise_types:
            self.noise_funcs['white'] = self._white_noise
        if 'pink' in self.noise_types:
            self.noise_funcs['pink'] = self._pink_noise
        if 'street' in self.noise_types:
            self.noise_funcs['street'] = self._street_noise
        if 'cafe' in self.noise_types:
            self.noise_funcs['cafe'] = self._cafe_noise
    
    def _white_noise(self, y, sr, snr_db):
        """Белый шум"""
        noise = np.random.normal(0, 0.01, len(y))
        return self._add_noise_with_snr(y, noise, snr_db)
    
    def _pink_noise(self, y, sr, snr_db):
        """Розовый шум"""
        samples = len(y)
        white = np.random.randn(samples)
        pink = np.zeros(samples)
        for i in range(1, samples):
            pink[i] = (pink[i-1] + 0.998 * white[i-1]) / 1.001
        pink = pink / (np.max(np.abs(pink)) + 1e-10)
        return self._add_noise_with_snr(y, pink, snr_db)
    
    def _street_noise(self, y, sr, snr_db):
        """Шум улицы (заглушка)"""
        noise = np.random.normal(0, 0.02, len(y))
        return self._add_noise_with_snr(y, noise, snr_db)
    
    def _cafe_noise(self, y, sr, snr_db):
        """Шум кафе (заглушка)"""
        noise = np.random.normal(0, 0.015, len(y))
        return self._add_noise_with_snr(y, noise, snr_db)
    
    def _add_noise_with_snr(self, signal, noise, snr_db):
        """Добавление шума с заданным SNR"""
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power > 0:
            snr_linear = 10 ** (snr_db / 10)
            noise_scaled = noise * np.sqrt(signal_power / (noise_power * snr_linear))
        else:
            noise_scaled = noise * 0.01
        
        return signal + noise_scaled
    
    def _apply_phone_filter(self, y, sr):
        """Имитация телефонного канала (300-3400 Гц)"""
        from scipy import signal
        
        b, a = signal.butter(4, [300 / (sr/2), 3400 / (sr/2)], btype='band')
        y_filtered = signal.filtfilt(b, a, y)
        y_filtered = np.tanh(y_filtered * 2) / 2
        
        if sr != 8000:
            y_filtered = librosa.resample(y_filtered, orig_sr=sr, target_sr=8000)
        
        return y_filtered
    
    def process_single_file(self, audio_path):
        """Обработка одного файла"""
        try:
            y, sr = librosa.load(audio_path, sr=16000, duration=5)
            if len(y) == 0:
                return 0
            
            y = y / (np.max(np.abs(y)) + 1e-10)
            
            base_name = audio_path.stem
            rel_path = audio_path.relative_to(self.input_dir)
            class_name = rel_path.parts[0]
            
            output_class_dir = self.output_dir / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)
            
            generated = 0
            
            # 1. Оригинал (всегда)
            y_8k = librosa.resample(y, orig_sr=sr, target_sr=8000)
            sf.write(output_class_dir / f"{base_name}_original.wav", y_8k, 8000)
            generated += 1
            
            # 2. Аугментация только для части файлов
            if random.random() <= self.augment_ratio:
                # Шумовые аугментации
                for noise_name, noise_func in self.noise_funcs.items():
                    for snr in self.snr_levels:
                        y_noisy = noise_func(y, sr, snr)
                        y_noisy = librosa.resample(y_noisy, orig_sr=sr, target_sr=8000)
                        sf.write(output_class_dir / f"{base_name}_noise_{noise_name}_snr{snr}.wav", 
                                y_noisy, 8000)
                        generated += 1
                
                # Телефонный фильтр
                if self.include_phone:
                    y_phone = self._apply_phone_filter(y, sr)
                    sf.write(output_class_dir / f"{base_name}_phone.wav", y_phone, 8000)
                    generated += 1
            
            return generated
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return 0
    
    def run(self):
        """Запуск аугментации"""
        audio_files = list(self.input_dir.rglob('*.wav')) + \
                      list(self.input_dir.rglob('*.mp3')) + \
                      list(self.input_dir.rglob('*.flac'))
        
        n_original = len(audio_files)
        print(f"\n📂 Found {n_original} original audio files")
        
        with Pool(processes=self.n_workers) as pool:
            results = list(tqdm(
                pool.imap(self.process_single_file, audio_files),
                total=n_original,
                desc="Processing"
            ))
        
        total_generated = sum(results)
        avg_per_file = total_generated / n_original if n_original > 0 else 0
        
        print(f"\n✅ Augmentation completed!")
        print(f"   Original files: {n_original}")
        print(f"   Generated files: {total_generated}")
        print(f"   Average per original: {avg_per_file:.1f}")
        print(f"   Output directory: {self.output_dir}")


def main():
    augmentor = AudioAugmentor(
        input_dir='data/audio',
        output_dir='data/processed/augmented_8khz',
        snr_levels=[10, 20],        # 2 уровня SNR
        noise_types=['white', 'pink'], # 2 типа шума
        augment_ratio=0.5,          # аугментировать 50% файлов
        include_phone=True,         # телефонный фильтр
        n_workers=None              # использовать все ядра
    )
    augmentor.run()


if __name__ == "__main__":
    main()