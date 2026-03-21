"""
Извлечение фонетических признаков для детекции синтезированной речи
Путь: src/features/phonetic_features.py
"""

import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')


class PhoneticFeatureExtractor:
    """
    Класс для извлечения фонетических признаков
    Форманты, VOT, спектральные огибающие
    """
    
    def __init__(self, sample_rate=8000, n_fft=512, hop_length=256):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract_formants(self, y, sr):
        """
        Извлечение формант (F1, F2, F3) - частотные пики спектра
        Синтезированная речь часто имеет неестественные форманты
        """
        # Вычисляем спектрограмму
        D = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        
        # Находим пики спектра для каждого временного окна
        formants = []
        
        for i in range(D.shape[1]):
            spectrum = D[:, i]
            # Ищем локальные максимумы
            peaks = self._find_peaks(spectrum)
            
            if len(peaks) >= 3:
                # Берем первые три форманты (F1, F2, F3)
                formants.append(peaks[:3])
            else:
                formants.append([0, 0, 0])
        
        formants = np.array(formants)
        
        # Статистики формант
        features = []
        for i in range(3):
            features.append(np.mean(formants[:, i]))  # среднее
            features.append(np.std(formants[:, i]))   # стандартное отклонение
        
        return np.array(features)
    
    def _find_peaks(self, spectrum, min_distance=3):
        """Поиск локальных максимумов спектра"""
        peaks = []
        for i in range(min_distance, len(spectrum) - min_distance):
            if (spectrum[i] > spectrum[i-1] and 
                spectrum[i] > spectrum[i+1] and
                spectrum[i] > spectrum[i-2] and
                spectrum[i] > spectrum[i+2]):
                peaks.append(i)
        
        # Сортируем по амплитуде и берем топ-3
        if peaks:
            peaks_vals = [(p, spectrum[p]) for p in peaks]
            peaks_vals.sort(key=lambda x: x[1], reverse=True)
            return [p for p, _ in peaks_vals[:3]]
        return []
    
    def extract_vot(self, y, sr):
        """
        Voice Onset Time - время между взрывом согласного и началом голоса
        У синтезированной речи VOT часто слишком стабилен
        """
        # Огибающая сигнала
        envelope = np.abs(y)
        
        # Сглаживаем
        window_size = int(sr * 0.01)  # 10 мс
        if len(envelope) > window_size:
            envelope = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
        
        # Находим переходы (взрывы)
        derivative = np.diff(envelope)
        
        # Ищем резкие изменения
        threshold = np.std(derivative) * 2
        onsets = np.where(np.abs(derivative) > threshold)[0]
        
        if len(onsets) > 1:
            # VOT = время между onset и началом голоса
            vot_values = np.diff(onsets) / sr
            features = [
                np.mean(vot_values),      # средний VOT
                np.std(vot_values),       # вариативность VOT
                np.max(vot_values),       # максимальный VOT
                np.min(vot_values),       # минимальный VOT
                len(vot_values)           # количество переходов
            ]
        else:
            features = [0, 0, 0, 0, 0]
        
        return np.array(features)
    
    def extract_spectral_envelope(self, y, sr):
        """
        Спектральная огибающая - гладкость спектра
        Синтезированная речь часто имеет слишком гладкую огибающую
        """
        # Строим спектр
        spectrum = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), 1/sr)
        
        # Ограничиваем до 4 кГц (телефонный диапазон)
        mask = freqs <= 4000
        spectrum = spectrum[mask]
        freqs = freqs[mask]
        
        # Логарифмический масштаб
        log_spectrum = np.log(spectrum + 1e-10)
        
        # Вычисляем производную (неровность)
        derivative = np.diff(log_spectrum)
        
        features = [
            np.mean(log_spectrum),           # средний уровень
            np.std(log_spectrum),            # вариативность
            np.mean(np.abs(derivative)),     # средняя неровность
            np.std(derivative),              # вариативность неровности
            np.max(derivative),              # максимальный скачок
            np.min(derivative)               # минимальный скачок
        ]
        
        return np.array(features)
    
    def extract_harmonic_noise_ratio(self, y, sr):
        """
        Отношение гармонической составляющей к шумовой
        У синтезированной речи HNR часто слишком высок
        """
        # Разделяем на гармоническую и шумовую части
        harmonic = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)
        
        # Энергия
        energy_harmonic = np.sum(harmonic ** 2)
        energy_noise = np.sum(percussive ** 2)
        energy_total = energy_harmonic + energy_noise
        
        if energy_total > 0:
            hnr = energy_harmonic / energy_total
        else:
            hnr = 0
        
        # Вариативность HNR по времени
        frame_length = int(sr * 0.025)  # 25 мс
        hop_length = int(sr * 0.010)    # 10 мс
        
        hnr_frames = []
        for start in range(0, len(y) - frame_length, hop_length):
            frame = y[start:start + frame_length]
            harm = librosa.effects.harmonic(frame)
            perc = librosa.effects.percussive(frame)
            
            e_harm = np.sum(harm ** 2)
            e_noise = np.sum(perc ** 2)
            e_total = e_harm + e_noise
            
            if e_total > 0:
                hnr_frames.append(e_harm / e_total)
            else:
                hnr_frames.append(0)
        
        features = [
            hnr,                           # общий HNR
            np.mean(hnr_frames),           # средний по фреймам
            np.std(hnr_frames),            # вариативность HNR
            np.max(hnr_frames),            # максимальный
            np.min(hnr_frames)             # минимальный
        ]
        
        return np.array(features)
    
    def extract_all(self, audio_path):
        """
        Извлечение всех фонетических признаков
        Возвращает 27 фонетических признаков
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=5)
            
            if y is None or len(y) == 0:
                return None
            
            y = y / (np.max(np.abs(y)) + 1e-10)
            
            features = []
            
            # 1. Форманты (3*2 = 6 признаков)
            formants = self.extract_formants(y, sr)
            features.extend(formants)
            
            # 2. VOT (5 признаков)
            vot = self.extract_vot(y, sr)
            features.extend(vot)
            
            # 3. Спектральная огибающая (6 признаков)
            envelope = self.extract_spectral_envelope(y, sr)
            features.extend(envelope)
            
            # 4. HNR (5 признаков)
            hnr = self.extract_harmonic_noise_ratio(y, sr)
            features.extend(hnr)
            
            # 5. Дополнительные признаки (5 признаков)
            # Формантные соотношения
            if len(formants) >= 6:
                f1_mean = formants[0]
                f2_mean = formants[2]
                f3_mean = formants[4]
                features.append(f2_mean / (f1_mean + 1e-10))  # F2/F1
                features.append(f3_mean / (f2_mean + 1e-10))  # F3/F2
                features.append(f1_mean / (f2_mean + 1e-10))  # F1/F2
                features.append(f2_mean - f1_mean)            # F2-F1
                features.append(f3_mean - f2_mean)            # F3-F2
            else:
                features.extend([0, 0, 0, 0, 0])
            
            # Проверяем размерность (должно быть 27)
            if len(features) != 27:
                if len(features) < 27:
                    features.extend([0] * (27 - len(features)))
                else:
                    features = features[:27]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Ошибка извлечения фонетических признаков: {e}")
            return None


class CombinedFeatureExtractor:
    """
    Класс для объединения акустических и фонетических признаков
    """
    
    def __init__(self, acoustic_dim=38, phonetic_dim=27):
        self.acoustic_dim = acoustic_dim
        self.phonetic_dim = phonetic_dim
        self.total_dim = acoustic_dim + phonetic_dim
        
        self.phonetic_extractor = PhoneticFeatureExtractor()
    
    def extract_acoustic_features(self, audio_path):
        """Извлечение существующих акустических признаков (38)"""
        try:
            y, sr = librosa.load(audio_path, sr=8000, duration=5)
            
            if y is None or len(y) == 0:
                return None
            
            y = y / (np.max(np.abs(y)) + 1e-10)
            
            features = []
            
            # MFCC (26)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=256)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            features.extend(mfcc_mean)
            features.extend(mfcc_std)
            
            # Спектральные (3)
            try:
                features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=512, hop_length=256)))
            except:
                features.append(0)
            
            try:
                features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=512, hop_length=256)))
            except:
                features.append(0)
            
            try:
                features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=512, hop_length=256)))
            except:
                features.append(0)
            
            # ZCR (1)
            try:
                features.append(np.mean(librosa.feature.zero_crossing_rate(y, frame_length=512, hop_length=256)))
            except:
                features.append(0)
            
            # RMS (1)
            try:
                features.append(np.mean(librosa.feature.rms(y=y, frame_length=512, hop_length=256)))
            except:
                features.append(0)
            
            # Tempo (1)
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=256)
                if isinstance(tempo, np.ndarray):
                    tempo = tempo[0] if len(tempo) > 0 else 0
                features.append(float(tempo))
            except:
                features.append(0)
            
            # Chroma (6)
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=512, hop_length=256)
                chroma_mean = np.mean(chroma, axis=1)
                features.extend(chroma_mean[:6])
            except:
                features.extend([0, 0, 0, 0, 0, 0])
            
            if len(features) != 38:
                if len(features) < 38:
                    features.extend([0] * (38 - len(features)))
                else:
                    features = features[:38]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Ошибка извлечения акустических признаков: {e}")
            return None
    
    def extract_combined(self, audio_path):
        """Извлечение объединенных признаков (акустика + фонетика)"""
        acoustic = self.extract_acoustic_features(audio_path)
        phonetic = self.phonetic_extractor.extract_all(audio_path)
        
        if acoustic is None or phonetic is None:
            return None
        
        combined = np.concatenate([acoustic, phonetic])
        
        if len(combined) != self.total_dim:
            if len(combined) < self.total_dim:
                combined = np.pad(combined, (0, self.total_dim - len(combined)))
            else:
                combined = combined[:self.total_dim]
        
        return combined