#!/usr/bin/env python3
"""
Быстрый старт проекта с проверкой всех зависимостей
Путь: scripts/quick_start.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Быстрый старт"""
    print("="*60)
    print("🚀 БЫСТРЫЙ СТАРТ ПРОЕКТА")
    print("="*60)
    
    # Проверяем Python
    print(f"\n🔍 Python: {sys.version}")
    
    # Устанавливаем базовые зависимости
    print("\n📦 Установка базовых зависимостей...")
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'
    ])
    
    # Устанавливаем pyyaml (самый важный для начала)
    print("\n📦 Установка pyyaml...")
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', 'pyyaml'
    ])
    
    # Запускаем полную установку
    print("\n📦 Запуск полной установки зависимостей...")
    subprocess.run([
        sys.executable, 'scripts/install_dependencies.py'
    ])
    
    print("\n" + "="*60)
    print("✅ ПОДГОТОВКА ЗАВЕРШЕНА")
    print("="*60)
    print("\nТеперь можно запустить пайплайн:")
    print(f"  {sys.executable} scripts/run_full_pipeline.py")
    print("\nИли интерфейс ручной проверки:")
    print(f"  {sys.executable} -m streamlit run src/manual_check/app.py")


if __name__ == "__main__":
    main()