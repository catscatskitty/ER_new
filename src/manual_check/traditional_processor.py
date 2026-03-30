import numpy as np
import joblib
import librosa
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features.acoustic.extract_acoustic_features import AcousticFeatureExtractor


class TraditionalModelProcessor:
    def __init__(self, models_root):
        self.models_root = Path(models_root)
        self.models = {}
        self.scalers = {}
        
        self.acoustic_extractor = AcousticFeatureExtractor()
        self.load_models()
    
    def load_models(self):
        for name in ['logistic', 'random_forest', 'xgboost', 'catboost']:
            path = self.models_root / name / 'model.pkl'
            if path.exists():
                data = joblib.load(path)
                self.models[name] = data['model']
                self.scalers[name] = data.get('scaler')
                print(f"Loaded {name}")
    
    def classify_audio(self, audio_path):
        if not self.models:
            return None
        
        features = self.acoustic_extractor.extract_from_file(audio_path)
        if features is None:
            return None
        
        results = {'model_predictions': {}, 'human_votes': 0, 'robot_votes': 0, 'total_confidence': 0}
        
        for name, model in self.models.items():
            X = features.reshape(1, -1)
            if name in self.scalers and self.scalers[name]:
                X = self.scalers[name].transform(X)
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                pred = 0 if proba[0] > proba[1] else 1
                conf = max(proba)
            else:
                pred = model.predict(X)[0]
                conf = 1.0
            
            pred_label = 'human' if pred == 0 else 'robot'
            results['model_predictions'][name] = {'prediction': pred_label, 'confidence': float(conf)}
            
            if pred_label == 'human':
                results['human_votes'] += 1
            else:
                results['robot_votes'] += 1
            results['total_confidence'] += conf
        
        results['final_prediction'] = 'human' if results['human_votes'] > results['robot_votes'] else 'robot'
        results['average_confidence'] = results['total_confidence'] / len(self.models)
        
        return results