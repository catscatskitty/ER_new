"""
Обработка аудио трёхмодальной моделью (спектрограммы + MFCC + фонетика)
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import traceback

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.deep_learning.train_trimodal import TriModalModel
from src.features.inference_spectrogram import extract_spectrogram
from src.features.inference_mfcc import extract_mfcc_sequence
from src.linguistic.inference_phonetic import extract_phonetic_from_audio


class TriModalProcessor:
    def __init__(self, models_root):
        if isinstance(models_root, str):
            self.models_root = Path(models_root)
        else:
            self.models_root = models_root

        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = 8000

        print(f"\n=== Инициализация TriModalProcessor ===")
        print(f"Устройство: {self.device}")
        print(f"Поиск моделей в: {self.models_root}")
        self.load_models()

    def load_models(self):
        """Загрузка трёхмодальной модели"""
        print(f"\n--- Загрузка трёхмодальной модели ---")
        model_dir = self.models_root / 'trimodal'
        model_path = model_dir / 'best_trimodal.pth'

        if not model_path.exists():
            print(f"⚠️ Модель не найдена: {model_path}")
            return 0

        try:
            model = TriModalModel().to(self.device)
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

            # Извлекаем state_dict (может быть в поле 'model_state_dict')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict)
            model.eval()
            self.model = model
            print(f"  ✅ Трёхмодальная модель загружена")
            return 1
        except Exception as e:
            print(f"  ❌ Ошибка загрузки: {e}")
            traceback.print_exc()
            return 0

    def classify_audio(self, audio_path):
        if self.model is None:
            print("Нет загруженной трёхмодальной модели!")
            return None

        print("\n=== Начало классификации ===")

        # 1. Спектрограмма
        print("Извлечение спектрограммы...")
        try:
            spec = extract_spectrogram(audio_path, sample_rate=self.sample_rate)
            if spec is None:
                print("❌ Ошибка: спектрограмма не извлечена")
                return None
            print(f"✅ Спектрограмма: форма {spec.shape}")
        except Exception as e:
            print(f"❌ Ошибка при извлечении спектрограммы: {e}")
            traceback.print_exc()
            return None

        # 2. MFCC
        print("Извлечение MFCC...")
        try:
            mfcc = extract_mfcc_sequence(audio_path, sample_rate=self.sample_rate)
            if mfcc is None:
                print("❌ Ошибка: MFCC не извлечены")
                return None
            print(f"✅ MFCC: форма {mfcc.shape}")
        except Exception as e:
            print(f"❌ Ошибка при извлечении MFCC: {e}")
            traceback.print_exc()
            return None

        # 3. Фонетика
        print("Извлечение фонетических признаков...")
        try:
            phon = extract_phonetic_from_audio(audio_path, sample_rate=self.sample_rate)
            if phon is None:
                print("❌ Ошибка: фонетические признаки не извлечены")
                return None
            print(f"✅ Фонетика: форма {phon.shape}")
        except Exception as e:
            print(f"❌ Ошибка при извлечении фонетики: {e}")
            traceback.print_exc()
            return None

        # Преобразуем в тензоры и добавляем batch dimension
        try:
            spec_t = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,128,128)
            mfcc_t = torch.FloatTensor(mfcc).unsqueeze(0).to(self.device)                # (1,128,13)
            phon_t = torch.FloatTensor(phon).unsqueeze(0).to(self.device)                # (1,27)
            print("✅ Тензоры созданы")
        except Exception as e:
            print(f"❌ Ошибка создания тензоров: {e}")
            traceback.print_exc()
            return None

        # Инференс
        try:
            with torch.no_grad():
                outputs = self.model(spec_t, mfcc_t, phon_t)
                probs = F.softmax(outputs, dim=1)[0]
                pred_class = torch.argmax(probs).item()
                confidence = probs[pred_class].item()
                pred_label = 'human' if pred_class == 0 else 'robot'
                print(f"✅ Инференс выполнен: {pred_label} ({confidence:.2%})")
        except Exception as e:
            print(f"❌ Ошибка инференса: {e}")
            traceback.print_exc()
            return None

        # Формируем результат
        results = {
            'model_predictions': {
                'trimodal': {
                    'prediction': pred_label,
                    'confidence': confidence,
                    'class': pred_class,
                    'probabilities': probs.cpu().numpy().tolist()
                }
            },
            'human_votes': 1 if pred_label == 'human' else 0,
            'robot_votes': 1 if pred_label == 'robot' else 0,
            'total_confidence': confidence,
            'final_prediction': pred_label,
            'average_confidence': confidence
        }
        return results