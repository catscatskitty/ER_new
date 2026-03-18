"""
Обработка аудио традиционными ML моделями
"""

import numpy as np
import joblib
import librosa
from pathlib import Path
import sys
import traceback

sys.path.append(str(Path(__file__).parent.parent.parent))


class TraditionalModelProcessor:
    def __init__(self, models_root):
        if isinstance(models_root, str):
            self.models_root = Path(models_root)
        else:
            self.models_root = models_root
            
        self.models = {}
        self.scalers = {}
        self.sample_rate = 8000
        self.feature_dim = 38
        
        print(f"\n=== Инициализация TraditionalModelProcessor ===")
        print(f"Поиск моделей в: {self.models_root}")
    
    def extract_features(self, audio_path):
        """Извлечение 38 признаков из аудио"""
        try:
            if isinstance(audio_path, str):
                audio_path = Path(audio_path)
            
            if not audio_path.exists():
                print(f"Файл не существует: {audio_path}")
                return None
            
            y, sr = librosa.load(str(audio_path), sr=self.sample_rate, duration=5)
            
            if y is None or len(y) == 0:
                return None
            
            y = y / (np.max(np.abs(y)) + 1e-10)
            
            features = []
            
            # 1. MFCC (13 means + 13 stds = 26 признаков)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=256)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            features.extend(mfcc_mean)
            features.extend(mfcc_std)
            
            # 2. Спектральные признаки (3)
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
            
            # 3. ZCR (1)
            try:
                features.append(np.mean(librosa.feature.zero_crossing_rate(y, frame_length=512, hop_length=256)))
            except:
                features.append(0)
            
            # 4. RMS (1)
            try:
                features.append(np.mean(librosa.feature.rms(y=y, frame_length=512, hop_length=256)))
            except:
                features.append(0)
            
            # 5. Tempo (1)
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=256)
                if isinstance(tempo, np.ndarray):
                    tempo = tempo[0] if len(tempo) > 0 else 0
                features.append(float(tempo))
            except:
                features.append(0)
            
            # 6. Chroma (6 признаков) - ИСПРАВЛЕНО: всегда 6
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=512, hop_length=256)
                chroma_mean = np.mean(chroma, axis=1)
                # Берем первые 6, но если меньше - дополняем
                if len(chroma_mean) >= 6:
                    features.extend(chroma_mean[:6])
                else:
                    features.extend(chroma_mean)
                    features.extend([0] * (6 - len(chroma_mean)))
            except:
                features.extend([0, 0, 0, 0, 0, 0])
            
            # Проверяем длину
            if len(features) != 38:
                print(f"⚠️ Ошибка: получено {len(features)} признаков, ожидалось 38")
                # Принудительно приводим к 38
                if len(features) < 38:
                    features.extend([0] * (38 - len(features)))
                else:
                    features = features[:38]
            
            return np.array(features, dtype=np.float32).reshape(1, -1)
            
        except Exception as e:
            print(f"Ошибка при извлечении признаков: {e}")
            return None
    
    def load_models(self):
        """Загрузка традиционных моделей"""
        print(f"\n--- Загрузка традиционных моделей ---")
        
        model_configs = [
            ('logistic', 'logistic/model.pkl'),
            ('random_forest', 'random_forest/model.pkl'),
            ('xgboost', 'xgboost/model.pkl'),
            ('catboost', 'catboost/model.pkl')
        ]
        
        loaded = 0
        
        for model_name, model_path in model_configs:
            full_path = self.models_root / model_path
            
            if not full_path.exists():
                print(f"⚠️ Модель не найдена: {full_path}")
                continue
            
            try:
                data = joblib.load(full_path)
                
                if isinstance(data, dict):
                    model = data.get('model')
                    scaler = data.get('scaler')
                else:
                    model = data
                    scaler = None
                
                if model is not None:
                    self.models[model_name] = model
                    self.scalers[model_name] = scaler
                    loaded += 1
                    print(f"✅ Загружена традиционная модель: {model_name}")
                    
                    if scaler is not None:
                        expected = scaler.mean_.shape[0]
                        if expected != 38:
                            print(f"   ⚠️ Размерность: {expected} (ожидалось 38)")
                        else:
                            print(f"   ✅ Размерность: {expected}")
                
            except Exception as e:
                print(f"❌ Ошибка загрузки {model_name}: {e}")
                traceback.print_exc()
        
        print(f"\n✅ Загружено традиционных моделей: {loaded}")
        return loaded
    
    def classify_audio(self, audio_path):
        if not self.models:
            print("Нет загруженных традиционных моделей!")
            return None
        
        features = self.extract_features(audio_path)
        if features is None:
            return None
        
        results = {
            'model_predictions': {},
            'human_votes': 0,
            'robot_votes': 0,
            'total_confidence': 0
        }
        
        for model_name, model in self.models.items():
            try:
                X = features
                
                # Применяем скейлер если есть
                if model_name in self.scalers and self.scalers[model_name] is not None:
                    scaler = self.scalers[model_name]
                    # Проверяем размерность
                    expected = scaler.mean_.shape[0]
                    if X.shape[1] != expected:
                        print(f"❌ Критическая ошибка: {model_name} ожидает {expected} признаков, получено {X.shape[1]}")
                        continue
                    
                    X = scaler.transform(X)
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[0]
                    pred_class = np.argmax(proba)
                    confidence = proba[pred_class]
                else:
                    pred_class = model.predict(X)[0]
                    confidence = 1.0
                
                pred_label = 'human' if pred_class == 0 else 'robot'
                
                results['model_predictions'][model_name] = {
                    'prediction': pred_label,
                    'confidence': float(confidence),
                    'class': int(pred_class),
                    'probabilities': proba.tolist() if 'proba' in locals() else [1-confidence, confidence]
                }
                
                if pred_label == 'human':
                    results['human_votes'] += 1
                else:
                    results['robot_votes'] += 1
                
                results['total_confidence'] += confidence
                
            except Exception as e:
                print(f"Ошибка при предсказании {model_name}: {e}")
        
        if results['human_votes'] > results['robot_votes']:
            results['final_prediction'] = 'human'
        elif results['robot_votes'] > results['human_votes']:
            results['final_prediction'] = 'robot'
        else:
            avg_human = self._get_average_confidence(results, 'human')
            avg_robot = self._get_average_confidence(results, 'robot')
            results['final_prediction'] = 'human' if avg_human > avg_robot else 'robot'
        
        total = len(results['model_predictions'])
        if total > 0:
            results['average_confidence'] = results['total_confidence'] / total
        else:
            results['average_confidence'] = 0
        
        return results
    
    def _get_average_confidence(self, results, target_class):
        confidences = [pred['confidence'] for pred in results['model_predictions'].values() if pred['prediction'] == target_class]
        return np.mean(confidences) if confidences else 0