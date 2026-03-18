#!/usr/bin/env python
"""
Очистка результатов экспериментов
"""

import os
import shutil
from pathlib import Path
import argparse

def clean_results(confirm: bool = True):
    """
    Очистка папки с результатами
    """
    results_dir = Path('results')
    
    if not results_dir.exists():
        print("Папка results не найдена")
        return
    
    print("=" * 60)
    print("ОЧИСТКА РЕЗУЛЬТАТОВ")
    print("=" * 60)
    
    # Подсчет файлов для удаления
    total_files = 0
    for root, dirs, files in os.walk(results_dir):
        total_files += len(files)
    
    print(f"Будет удалено {total_files} файлов в {results_dir}")
    
    if confirm:
        response = input("Продолжить? (y/n): ")
        if response.lower() != 'y':
            print("Операция отменена")
            return
    
    # Удаление
    try:
        shutil.rmtree(results_dir)
        print(f"✅ Папка {results_dir} удалена")
        
        # Создание заново
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / 'trained_models').mkdir(exist_ok=True)
        (results_dir / 'metrics').mkdir(exist_ok=True)
        (results_dir / 'plots').mkdir(exist_ok=True)
        (results_dir / 'manual_checks').mkdir(exist_ok=True)
        
        print("✅ Структура папок восстановлена")
        
    except Exception as e:
        print(f"❌ Ошибка при очистке: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='Принудительная очистка без подтверждения')
    args = parser.parse_args()
    
    clean_results(confirm=not args.force)