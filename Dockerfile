FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей, необходимых для звуковых библиотек и ML
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода и артефактов
COPY src src/
COPY api.py .
COPY app.py .
COPY data/models/ml/ data/models/ml/

# Команда запуска: Установили порт 8080 для обхода конфликтов с портом 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
