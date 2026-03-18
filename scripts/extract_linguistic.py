#!/usr/bin/env python
"""
Только извлечение лингвистических признаков
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def extract_linguistic(config_path: str = "configs/linguistic_config.yaml"):
    """
    Запуск только извлечения лингвистических признаков
    """
    print("=" * 60)
    print("ИЗВЛЕЧЕНИЕ ЛИНГВИСТИЧЕСКИХ ПРИЗНАКОВ")
    print("=" * 60)
    
    # Проверка наличия транскрипций
    transcripts_dir = Path('data/transcripts')
    if not transcripts_dir.exists():
        print("❌ Директория с транскрипциями не найдена")
        print("Сначала запустите транскрибацию:")
        print("  python src/linguistic/transcriber.py")
        return False
    
    # Запуск извлечения признаков
    cmd = [sys.executable, 'src/linguistic/pipeline/extract_linguistic_features.py', 
           '--config', config_path]
    
    try:
        subprocess.run(cmd)
        return True
    except Exception as e:
        print(f"Ошибка при извлечении признаков: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/linguistic_config.yaml')
    args = parser.parse_args()
    
    extract_linguistic(args.config)