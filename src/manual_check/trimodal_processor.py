import numpy as np
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.deep_learning.trimodal import TriModalModel
from src.features.spectrogram.extract_spectrogram import SpectrogramExtractor
from src.features.mfcc.extract_mfcc_sequence import MFCCSequenceExtractor
from src.features.phonetic.extract_phonetic_features import PhoneticFeatureExtractor


class TriModalProcessor:
    def __init__(self, models_root):
        self.models_root = Path(models_root)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.spec_extractor = SpectrogramExtractor()
        self.mfcc_extractor = MFCCSequenceExtractor()
        self.phonetic_extractor = PhoneticFeatureExtractor()
        
        self.load_models()
    
    def load_models(self):
        model_path = self.models_root / 'trimodal' / 'best_trimodal.pth'
        norm_path = self.models_root / 'trimodal' / 'trimodal_normalization.npy'
        
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            return
        
        norm_data = np.load(norm_path, allow_pickle=True).item()
        self.spec_mean, self.spec_std = norm_data['spec']['mean'], norm_data['spec']['std']
        self.mfcc_mean, self.mfcc_std = norm_data['mfcc']['mean'], norm_data['mfcc']['std']
        self.phon_mean, self.phon_std = norm_data['phon']['mean'], norm_data['phon']['std']
        
        self.model = TriModalModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        print("TriModal model loaded")
    
    def classify_audio(self, audio_path):
        if not hasattr(self, 'model'):
            return None
        
        spec = self.spec_extractor.extract_from_file(audio_path)
        mfcc = self.mfcc_extractor.extract_from_file(audio_path)
        phon = self.phonetic_extractor.extract_all(audio_path)
        
        if spec is None or mfcc is None or phon is None:
            return None
        
        spec = (spec - self.spec_mean) / (self.spec_std + 1e-8)
        mfcc = (mfcc - self.mfcc_mean) / (self.mfcc_std + 1e-8)
        phon = (phon - self.phon_mean) / (self.phon_std + 1e-8)
        
        spec_t = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0).to(self.device)
        mfcc_t = torch.FloatTensor(mfcc).unsqueeze(0).to(self.device)
        phon_t = torch.FloatTensor(phon).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(spec_t, mfcc_t, phon_t)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_class = torch.argmax(probs).item()
            confidence = probs[pred_class].item()
            pred_label = 'human' if pred_class == 0 else 'robot'
        
        return {
            'model_predictions': {'trimodal': {'prediction': pred_label, 'confidence': confidence}},
            'human_votes': 1 if pred_label == 'human' else 0,
            'robot_votes': 1 if pred_label == 'robot' else 0,
            'total_confidence': confidence,
            'final_prediction': pred_label,
            'average_confidence': confidence
        }