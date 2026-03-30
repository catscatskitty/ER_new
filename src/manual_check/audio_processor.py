import numpy as np
import torch
import librosa
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.deep_learning.hybrid_spectrogram import HybridSpectrogram
from src.features.spectrogram.extract_spectrogram import SpectrogramExtractor


class AudioProcessor:
    def __init__(self, models_root, use_phonetic=False):
        self.models_root = Path(models_root)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_phonetic = use_phonetic  # игнорируется для гибридной модели
        
        self.spec_extractor = SpectrogramExtractor()
        self.load_models()
    
    def load_models(self):
        model_path = self.models_root / 'torch_models' / 'best_hybrid_spectrogram.pth'
        norm_path = self.models_root / 'torch_models' / 'hybrid_normalization.npy'
        
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            return
        
        norm_data = np.load(norm_path, allow_pickle=True).item()
        self.mean, self.std = norm_data['mean'], norm_data['std']
        
        self.model = HybridSpectrogram().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        print("Hybrid model loaded")
    
    def classify_audio(self, audio_path):
        if not hasattr(self, 'model'):
            return None
        
        spec = self.spec_extractor.extract_from_file(audio_path)
        if spec is None:
            return None
        
        spec = (spec - self.mean) / (self.std + 1e-8)
        spec_tensor = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(spec_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_class = torch.argmax(probs).item()
            confidence = probs[pred_class].item()
            pred_label = 'human' if pred_class == 0 else 'robot'
        
        return {
            'model_predictions': {'hybrid': {'prediction': pred_label, 'confidence': confidence}},
            'human_votes': 1 if pred_label == 'human' else 0,
            'robot_votes': 1 if pred_label == 'robot' else 0,
            'total_confidence': confidence,
            'final_prediction': pred_label,
            'average_confidence': confidence
        }