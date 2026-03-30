import numpy as np
import joblib
from pathlib import Path


class TraditionalPredictor:
    """Класс для инференса традиционных моделей"""
    
    def __init__(self, models_root='results/trained_models'):
        self.models_root = Path(models_root)
        self.models = {}
        self.scalers = {}
        self.load_models()
    
    def load_models(self):
        """Загрузка всех традиционных моделей"""
        for name in ['logistic', 'random_forest', 'xgboost', 'catboost']:
            path = self.models_root / name / 'model.pkl'
            if path.exists():
                data = joblib.load(path)
                self.models[name] = data['model']
                self.scalers[name] = data.get('scaler')
                print(f"Loaded {name}")
    
    def predict(self, features):
        """Предсказание для одного образца"""
        if not self.models:
            return None
        
        results = {}
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
            
            results[name] = {
                'prediction': 'human' if pred == 0 else 'robot',
                'confidence': float(conf),
                'class': int(pred)
            }
        
        # Голосование
        human_votes = sum(1 for r in results.values() if r['prediction'] == 'human')
        robot_votes = len(results) - human_votes
        
        final = 'human' if human_votes > robot_votes else 'robot'
        avg_conf = np.mean([r['confidence'] for r in results.values()])
        
        return {
            'model_predictions': results,
            'final_prediction': final,
            'average_confidence': avg_conf,
            'human_votes': human_votes,
            'robot_votes': robot_votes
        }