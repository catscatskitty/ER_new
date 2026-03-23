"""
Обработка аудио нейросетевыми моделями
Поддерживает акустические (38) и комбинированные (38+27) признаки.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from pathlib import Path
import sys
import traceback

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.linguistic.inference_phonetic import extract_phonetic_from_audio


# ==================== МОДЕЛИ ====================
class CNN1D(nn.Module):
    def __init__(self, input_dim=38, num_classes=2, dropout=0.5):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout/2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout/2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)          # (batch, 1, features)
        x = self.conv_layers(x)      # (batch, 256, 1)
        x = x.squeeze(-1)            # (batch, 256)
        return self.classifier(x)


class LSTMModel(nn.Module):
    def __init__(self, input_size=38, hidden_size=128, num_layers=2, num_classes=2, dropout=0.3, bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.projection = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)                # (batch, 1, features)
        x = self.projection(x)            # (batch, 1, hidden_size)
        lstm_out, (hidden, _) = self.lstm(x)

        if self.bidirectional:
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            hidden_concat = torch.cat((hidden_forward, hidden_backward), dim=1)
            out = hidden_concat
        else:
            out = hidden[-1, :, :]

        return self.classifier(out)


class HybridModel(nn.Module):
    def __init__(self, input_dim=38, hidden_size=64, num_classes=2, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)          # (batch, 1, features)
        x = self.cnn(x)              # (batch, 64, 1)
        x = x.squeeze(-1)            # (batch, 64)
        x = x.unsqueeze(1)           # (batch, 1, 64)
        lstm_out, (hidden, _) = self.lstm(x)
        hidden_concat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(hidden_concat)


# ==================== AUDIO PROCESSOR ====================
class AudioProcessor:
    def __init__(self, models_root, use_phonetic=False):
        if isinstance(models_root, str):
            self.models_root = Path(models_root)
        else:
            self.models_root = models_root

        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = 8000
        self.use_phonetic = use_phonetic
        self.acoustic_dim = 38
        self.phonetic_dim = 27
        self.feature_dim = self.acoustic_dim + (self.phonetic_dim if use_phonetic else 0)

        print(f"\n=== Инициализация AudioProcessor ===")
        print(f"Устройство: {self.device}")
        print(f"Фонетические признаки: {'включены' if use_phonetic else 'выключены'}")
        print(f"Размерность признаков: {self.feature_dim}")
        print(f"Поиск моделей в: {self.models_root}")
        self.load_models()

    def _extract_acoustic(self, audio_path):
        """Извлечение 38 акустических признаков"""
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
            features = np.zeros(self.acoustic_dim, dtype=np.float32)
            idx = 0

            # MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=256)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            features[idx:idx+13] = mfcc_mean[:13]
            idx += 13
            features[idx:idx+13] = mfcc_std[:13]
            idx += 13

            # Спектральные признаки (3)
            try:
                features[idx] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=512, hop_length=256))
            except:
                features[idx] = 0
            idx += 1
            try:
                features[idx] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=512, hop_length=256))
            except:
                features[idx] = 0
            idx += 1
            try:
                features[idx] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=512, hop_length=256))
            except:
                features[idx] = 0
            idx += 1

            # ZCR (1)
            try:
                features[idx] = np.mean(librosa.feature.zero_crossing_rate(y, frame_length=512, hop_length=256))
            except:
                features[idx] = 0
            idx += 1

            # RMS (1)
            try:
                features[idx] = np.mean(librosa.feature.rms(y=y, frame_length=512, hop_length=256))
            except:
                features[idx] = 0
            idx += 1

            # Tempo (1)
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=256)
                if isinstance(tempo, np.ndarray):
                    tempo = tempo[0] if len(tempo) > 0 else 0
                features[idx] = float(tempo)
            except:
                features[idx] = 0
            idx += 1

            # Chroma (6)
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=512, hop_length=256)
                chroma_mean = np.mean(chroma, axis=1)
                features[idx:idx+6] = chroma_mean[:6]
            except:
                pass
            idx += 6

            return features
        except Exception as e:
            print(f"Ошибка при извлечении акустических признаков: {e}")
            return None

    def extract_features(self, audio_path):
        """Извлечение полного набора признаков (акустика + опционально фонетика)"""
        acoustic = self._extract_acoustic(audio_path)
        if acoustic is None:
            return None
        if self.use_phonetic:
            phonetic = extract_phonetic_from_audio(audio_path, self.sample_rate)
            if phonetic is None:
                print("⚠️ Не удалось извлечь фонетические признаки, используем только акустику")
                return acoustic
            combined = np.concatenate([acoustic, phonetic])
            return combined
        else:
            return acoustic

    def load_models(self):
        print(f"\n--- Загрузка нейросетевых моделей ---")
        model_types = ['cnn', 'lstm', 'hybrid']
        torch_models_dir = self.models_root / 'torch_models'

        if not torch_models_dir.exists():
            print(f"⚠️ Директория не найдена: {torch_models_dir}")
            return 0

        loaded = 0
        for model_type in model_types:
            model_path = torch_models_dir / f'best_{model_type}.pth'
            if not model_path.exists():
                print(f"⚠️ Модель не найдена: {model_path}")
                continue
            try:
                if model_type == 'cnn':
                    model = CNN1D(input_dim=self.feature_dim, num_classes=2)
                elif model_type == 'lstm':
                    model = LSTMModel(input_size=self.feature_dim, num_classes=2)
                else:
                    model = HybridModel(input_dim=self.feature_dim, num_classes=2)

                # Загружаем чекпоинт
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

                # Проверяем структуру: если есть ключ 'model_state_dict', используем его
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint  # предполагаем, что это напрямую state_dict

                model.load_state_dict(state_dict)
                model = model.to(self.device)
                model.eval()
                self.models[model_type] = model
                loaded += 1
                print(f"  ✅ {model_type} загружена")
            except Exception as e:
                print(f"  ❌ Ошибка загрузки {model_type}: {e}")
                traceback.print_exc()

        print(f"\n✅ Загружено нейросетевых моделей: {loaded}")
        return loaded

    def classify_audio(self, audio_path):
        if not hasattr(self, 'model') or self.model is None:
            print("Нет загруженной трёхмодальной модели!")
            return None

        print("Извлечение спектрограммы...")
        spec = extract_spectrogram(audio_path, sample_rate=self.sample_rate)
        print("Извлечение MFCC...")
        mfcc = extract_mfcc_sequence(audio_path, sample_rate=self.sample_rate)
        print("Извлечение фонетики...")
        phon = extract_phonetic_from_audio(audio_path, sample_rate=self.sample_rate)
        print("Признаки извлечены")
        if spec is None or mfcc is None or phon is None:
            print(f"Ошибка извлечения: spec={spec is None}, mfcc={mfcc is None}, phon={phon is None}")
            return None
        
        if not self.models:
            print("Нет загруженных нейросетевых моделей!")
            return None

        features = self.extract_features(audio_path)
        if features is None:
            return None

        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        results = {
            'model_predictions': {},
            'human_votes': 0,
            'robot_votes': 0,
            'total_confidence': 0
        }

        for model_name, model in self.models.items():
            try:
                with torch.no_grad():
                    outputs = model(features_tensor)
                    probs = F.softmax(outputs, dim=1)[0]
                    pred_class = torch.argmax(probs).item()
                    confidence = probs[pred_class].item()
                    pred_label = 'human' if pred_class == 0 else 'robot'

                    results['model_predictions'][model_name] = {
                        'prediction': pred_label,
                        'confidence': confidence,
                        'class': pred_class,
                        'probabilities': probs.cpu().numpy().tolist()
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