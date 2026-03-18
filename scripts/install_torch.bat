@echo off
echo ========================================
echo УСТАНОВКА PYTORCH С ПОДДЕРЖКОЙ GPU
echo ========================================
echo.

:: Проверка наличия Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python не найден. Установите Python 3.8+
    pause
    exit /b 1
)

:: Проверка наличия CUDA
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [WARNING] NVIDIA GPU не обнаружена или не установлены драйверы
    echo Устанавливаем CPU версию PyTorch...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
) else (
    echo [OK] NVIDIA GPU обнаружена
    echo.
    echo Выберите версию CUDA:
    echo 1. CUDA 11.8 (рекомендуется)
    echo 2. CUDA 12.1
    echo 3. CPU only
    echo.
    set /p choice="Ваш выбор (1-3): "
    
    if "%choice%"=="1" (
        echo Установка PyTorch с CUDA 11.8...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ) else if "%choice%"=="2" (
        echo Установка PyTorch с CUDA 12.1...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ) else (
        echo Установка CPU версии PyTorch...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    )
)

:: Проверка установки
echo.
echo Проверка установки...
python -c "import torch; print('PyTorch версия:', torch.__version__); print('CUDA доступна:', torch.cuda.is_available())"

echo.
echo Установка завершена!
pause