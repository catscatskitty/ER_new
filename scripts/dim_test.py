# check_model_dimensions.py
import joblib
from pathlib import Path

models_root = Path('results/trained_models')

# Проверка традиционных моделей
for model_name in ['logistic', 'random_forest', 'xgboost', 'catboost']:
    model_path = models_root / model_name / 'model.pkl'
    if model_path.exists():
        data = joblib.load(model_path)
        scaler = data.get('scaler') if isinstance(data, dict) else None
        if scaler:
            print(f"{model_name}: ожидает {scaler.mean_.shape[0]} признаков")
        else:
            print(f"{model_name}: скейлер не найден")