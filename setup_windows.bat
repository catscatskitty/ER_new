@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo УСТАНОВКА СИСТЕМЫ "ЧЕЛОВЕК vs РОБОТ"
echo ========================================
echo.

:: Проверка Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python не найден!
    echo Скачайте и установите Python 3.8+ с https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Создание виртуального окружения
echo [1/7] Создание виртуального окружения...
if not exist venv (
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Не удалось создать виртуальное окружение
        pause
        exit /b 1
    )
)

:: Активация виртуального окружения
echo [2/7] Активация виртуального окружения...
call venv\Scripts\activate.bat

:: Обновление pip
echo [3/7] Обновление pip...
python -m pip install --upgrade pip

:: Установка зависимостей
echo [4/7] Установка зависимостей...
pip install -r requirements.txt

:: Установка PyTorch (выбор версии)
echo.
echo [5/7] Настройка PyTorch...
echo Выберите версию PyTorch:
echo 1. CPU (рекомендуется для начала)
echo 2. CUDA 11.8 (для NVIDIA GPU)
echo 3. CUDA 12.1 (для новых NVIDIA GPU)
echo.

set /p torch_choice="Ваш выбор (1-3): "

if "%torch_choice%"=="2" (
    echo Установка PyTorch с CUDA 11.8...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else if "%torch_choice%"=="3" (
    echo Установка PyTorch с CUDA 12.1...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo Установка CPU версии PyTorch...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

:: Загрузка моделей для NLP
echo [6/7] Загрузка моделей для NLP...
python -m spacy download ru_core_news_lg
python -c "import nltk; nltk.download('punkt')"

:: Создание структуры папок
echo [7/7] Создание структуры папок...
mkdir data\audio\human 2>nul
mkdir data\audio\robot 2>nul
mkdir data\processed 2>nul
mkdir data\transcripts 2>nul
mkdir data\splits 2>nul
mkdir results\trained_models 2>nul
mkdir results\metrics 2>nul
mkdir results\plots 2>nul
mkdir results\manual_checks 2>nul

echo.
echo ========================================
echo УСТАНОВКА ЗАВЕРШЕНА!
echo ========================================
echo.
echo Для запуска полного пайплайна:
echo   python scripts/run_full_pipeline.py
echo.
echo Для запуска интерфейса:
echo   streamlit run src/manual_check/app.py
echo.
pause