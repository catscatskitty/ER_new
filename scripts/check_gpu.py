#!/usr/bin/env python3
"""
Скрипт для проверки GPU
Путь: scripts/check_gpu.py
"""

import sys
from pathlib import Path

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.gpu_utils import (
    configure_gpu, configure_mixed_precision,
    print_gpu_info, set_random_seeds,
    get_device, clear_memory
)
from src.utils.config_loader import ConfigLoader


def main():
    print("="*60)
    print("🔍 ПРОВЕРКА GPU")
    print("="*60)
    
    # Проверяем PyTorch
    try:
        import torch
        print(f"✅ PyTorch версия: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch не установлен!")
        print("Установите: pip install torch torchvision torchaudio")
        return
    
    # Загружаем конфиг
    try:
        config_loader = ConfigLoader('configs')
        training_config = config_loader.load_config('training_config')
        gpu_config = training_config['training'].get('gpu', {'enabled': True})
        
        print("\n📋 Настройки из конфига:")
        print(f"  enabled: {gpu_config.get('enabled', True)}")
        print(f"  device: {gpu_config.get('device', '0')}")
        print(f"  mixed_precision: {gpu_config.get('mixed_precision', False)}")
        
    except Exception as e:
        print(f"\n⚠️ Не удалось загрузить конфиг: {e}")
        gpu_config = {'enabled': True, 'device': '0'}
    
    # Настраиваем GPU
    print("\n🔧 Настройка GPU:")
    gpu_available = configure_gpu(gpu_config)
    
    if gpu_available:
        # Информация о GPU
        print_gpu_info()
        
        # Mixed precision
        if 'training_config' in locals():
            configure_mixed_precision(training_config)
        
        # Тест производительности
        print("\n⏱️  Тест производительности:")
        device = get_device()
        
        # Создаем тензоры
        a = torch.randn(5000, 5000).to(device)
        b = torch.randn(5000, 5000).to(device)
        
        # Прогрев
        torch.matmul(a, b)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Замер
        import time
        start = time.time()
        for _ in range(10):
            c = torch.matmul(a, b)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"  ✅ 10 умножений матриц 5000x5000: {elapsed:.3f} сек")
        
        # Очищаем память
        clear_memory()
    else:
        print("\n💻 Используется CPU")
    
    print("\n" + "="*60)
    print("✅ ПРОВЕРКА ЗАВЕРШЕНА")
    print("="*60)


if __name__ == "__main__":
    main()