import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

class VotingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ансамбль с голосованием
    """
    
    def __init__(self, models=None, weights=None, voting='soft'):
        self.models = models or []
        self.weights = weights or [1] * len(self.models)
        self.voting = voting
        
    def fit(self, X, y):
        # Модели уже должны быть обучены
        return self
    
    def predict(self, X):
        if self.voting == 'soft':
            # Усреднение вероятностей
            probas = self.predict_proba(X)
            return np.argmax(probas, axis=1)
        else:
            # Жесткое голосование
            predictions = np.array([model.predict(X) for model in self.models])
            # Взвешенное голосование
            weighted_preds = np.average(predictions, axis=0, weights=self.weights)
            return np.round(weighted_preds).astype(int)
    
    def predict_proba(self, X):
        # Усреднение вероятностей с весами
        probas = []
        total_weight = sum(self.weights)
        
        for model, weight in zip(self.models, self.weights):
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
            else:
                # Для моделей без predict_proba
                pred = model.predict(X)
                proba = np.zeros((len(X), 2))
                proba[np.arange(len(X)), pred] = 1
            
            probas.append(proba * weight / total_weight)
        
        return np.sum(probas, axis=0)
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))