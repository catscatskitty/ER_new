import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import joblib

class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Стекинг ансамбль
    """
    
    def __init__(self, base_models=None, meta_model=None, cv=5):
        self.base_models = base_models or []
        self.meta_model = meta_model or LogisticRegression()
        self.cv = cv
        
    def fit(self, X, y):
        # Обучение базовых моделей
        for model in self.base_models:
            model.fit(X, y)
        
        # Генерация мета-признаков с кросс-валидацией
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        for i, model in enumerate(self.base_models):
            for train_idx, val_idx in skf.split(X, y):
                # Обучение на тренировочном фолде
                X_train_fold, y_train_fold = X[train_idx], y[train_idx]
                X_val_fold = X[val_idx]
                
                model_fold = model.__class__()
                model_fold.fit(X_train_fold, y_train_fold)
                
                # Предсказание на валидационном фолде
                if hasattr(model_fold, 'predict_proba'):
                    meta_features[val_idx, i] = model_fold.predict_proba(X_val_fold)[:, 1]
                else:
                    meta_features[val_idx, i] = model_fold.predict(X_val_fold)
        
        # Обучение мета-модели
        self.meta_model.fit(meta_features, y)
        
        # Дообучение базовых моделей на всех данных
        for model in self.base_models:
            model.fit(X, y)
        
        return self
    
    def predict(self, X):
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X):
        meta_features = self._get_meta_features(X)
        
        if hasattr(self.meta_model, 'predict_proba'):
            return self.meta_model.predict_proba(meta_features)
        else:
            pred = self.meta_model.predict(meta_features)
            proba = np.zeros((len(X), 2))
            proba[np.arange(len(X)), pred] = 1
            return proba
    
    def _get_meta_features(self, X):
        """Получение мета-признаков"""
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, model in enumerate(self.base_models):
            if hasattr(model, 'predict_proba'):
                meta_features[:, i] = model.predict_proba(X)[:, 1]
            else:
                meta_features[:, i] = model.predict(X)
        
        return meta_features
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))