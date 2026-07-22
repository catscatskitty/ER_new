import os
import json
from typing import Dict, Any

class Config:
    """
    Центральный класс для хранения всех констант и настроек проекта.
    Вместо использования "магических чисел" их здесь объявляются.
    """
    
    # --- Настройки Модели и ML ---
    MODEL = {
        "FEATURE_TYPE": "mfcc176", # Тип признака по умолчанию
        "ROBOT_WEIGHT": 1.5,       # Вес для класса "Робот" (из pipeline.py)
        "CLASSIFICATION_THRESHOLD": 0.70, # Порог принятия решения (из api.py и README)
        "N_WORKERS": 6             # Количество процессов для распараллеливания
    }
    
    # --- Пути к данным и моделям ---
    PATHS = {
        "RAW_DATA": "data/raw",
        "PROCESSED_DATA": "data/processed",
        "MODEL_DIR": "data/models/ml",
        "SCALER_PATH": "data/models/scaler.pkl"
    }

    @classmethod
    def load_from_json(cls, file_path: str = "settings.json"):
        """Попытка загрузить настройки из файла settings.json."""
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if 'threshold' in data:
                    cls.MODEL['CLASSIFICATION_THRESHOLD'] = data['threshold']
                if 'audio_device' in data:
                    # Здесь можно было бы хранить и другие системные настройки
                    pass
                print(f"Успешно загружены настройки из {file_path}.")
            except Exception as e:
                print(f"Предупреждение: Не удалось загрузить настройки из {file_path}: {e}")
        else:
            print(f"Предупреждение: Файл настроек {file_path} не найден.")

# Инициализация настроек при импорте
Config.load_from_json()