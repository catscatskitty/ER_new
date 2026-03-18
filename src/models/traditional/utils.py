"""
Вспомогательные функции для традиционных моделей
Путь: src/models/traditional/utils.py
"""

import numpy as np
import pickle
from pathlib import Path


def save_model(model, model_path, scaler=None, feature_names=None):
    """
    Сохранение модели и сопутствующих объектов
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_dict = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_dict, f)
    
    print(f"Модель сохранена: {model_path}")


def load_model(model_path):
    """
    Загрузка модели
    """
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
    
    return model_dict['model'], model_dict.get('scaler'), model_dict.get('feature_names')


def normalize_features(X_train, X_val=None, X_test=None):
    """
    Нормализация признаков (StandardScaler)
    """
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    result = [X_train_scaled]
    
    if X_val is not None:
        result.append(scaler.transform(X_val))
    if X_test is not None:
        result.append(scaler.transform(X_test))
    
    if len(result) == 1:
        return result[0], scaler
    else:
        return (*result, scaler)


def get_class_weights(y):
    """
    Вычисление весов классов для несбалансированных данных
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    return dict(zip(classes, weights))