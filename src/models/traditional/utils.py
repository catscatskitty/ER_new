import numpy as np
import joblib
from pathlib import Path


def save_model(model, scaler, model_path):
    """Сохранение модели и scaler"""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'scaler': scaler}, model_path)


def load_model(model_path):
    """Загрузка модели и scaler"""
    data = joblib.load(model_path)
    return data.get('model'), data.get('scaler')


def get_model_metrics(model, X_test, y_test):
    """Получение метрик модели"""
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    
    y_pred = model.predict(X_test)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }


def get_feature_importance(model, feature_names):
    """Получение важности признаков для моделей, поддерживающих это"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        return sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
        return sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    else:
        return None