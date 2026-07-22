import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np

# --- Импорты из src/ (Остаются как есть) ---
from src.utils import ScalerManager
from src.features import extract_features
from src.models_ml import MLModelSuite

# --- Глобальная Инициализация ---
# Логика загрузки ресурсов остается, но инициализация FastAPI становится минимальной.
try:
    # 1. Загрузка скалера
    scaler = ScalerManager("data/models/scaler.pkl").load()
    
    # 2. Инициализация MLSuite
    ml_suite = MLModelSuite(scaler_manager=scaler)
    print("API: Успешно инициализирован ModelHandler и ScalerManager. API готов к приему запросов.")
except Exception as e:
    print(f"API: КРИТИЧЕСКАЯ ОШИБКА ПРИ ИНИЦИАЛИЗАЦИИ ГЛОБАЛЬНЫХ РЕСУРСОВ: {e}")
    ml_suite = None
    scaler = None

# САМАЯ БЫСТРАЯ И НАДЕЖНАЯ ИНИЦИАЛИЗАЦИЯ API:
app = FastAPI()
# --- Эндпоинты API ---

@app.post("/predict")
def predict_endpoint(request: Request):
    """
    Принимает аудиофайл (multipart/form-data) и возвращает оценку подлинности.
    (ВАЖНО: Логика обработки файла должна быть здесь.)
    """
    from fastapi import UploadFile, File
    
    if 'audio' not in request.files:
        raise HTTPException(status_code=400, detail="Отсутствует аудиофайл 'audio' в запросе.")

    audio_file = request.files['audio']
    if audio_file.filename == '':
        raise HTTPException(status_code=400, detail="Не выбран аудиофайл.")

    # Логика обработки файла остаётся заглушкой, так как реальная обработка требует временных файлов.
    return JSONResponse({"status": "success", "message": "Endpoint /predict настроен. Требуется полная реализация обработки файла."})

@app.get("/info")
def info_endpoint():
    """
    Предоставляет информацию о версии и текущем состоянии.
    """
    return JSONResponse({"service": "VoiceShield API", "version": "1.0.0", "status": "File-based Backend Only"})

# ДОБАВЛЕН: Обработчик для корневого запроса, чтобы убрать ошибку ERR_CONNECTION_REFUSED
@app.get("/")
def read_root():
    """
    Главная точка входа. Информирует пользователя, что это API.
    """
    return JSONResponse({"message": "VoiceShield API запущен. Для работы используйте эндпоинты /predict или /health."})

