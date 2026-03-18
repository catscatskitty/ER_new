#!/usr/bin/env python
"""
Запуск интерфейса для ручной проверки
"""

import os
import sys
import subprocess
from pathlib import Path

def launch_manual_checker():
    """
    Запуск Streamlit приложения
    """
    # Определяем путь к главному приложению
    app_path = Path(__file__).parent.parent / 'src' / 'manual_check' / 'app.py'
    
    if not app_path.exists():
        print(f"Ошибка: Файл приложения не найден: {app_path}")
        return False
    
    print("=" * 60)
    print("ЗАПУСК ИНТЕРФЕЙСА РУЧНОЙ ПРОВЕРКИ")
    print("=" * 60)
    print(f"Приложение: {app_path}")
    print("\nПосле запуска откройте браузер по адресу: http://localhost:8501")
    print("Нажмите Ctrl+C для остановки\n")
    
    try:
        # Запуск Streamlit
        cmd = [sys.executable, '-m', 'streamlit', 'run', str(app_path)]
        subprocess.run(cmd)
        return True
    except KeyboardInterrupt:
        print("\nПриложение остановлено")
        return True
    except Exception as e:
        print(f"Ошибка при запуске: {e}")
        return False

if __name__ == "__main__":
    launch_manual_checker()