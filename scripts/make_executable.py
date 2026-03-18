#!/usr/bin/env python3
"""
Скрипт для установки прав на выполнение для Unix-систем
(На Windows просто пропускается)
Путь: scripts/make_executable.py
"""

import os
import sys
import stat
import platform
from pathlib import Path


def make_executable():
    """Установка прав на выполнение для Unix-систем"""
    
    if platform.system() == 'Windows':
        print("✅ На Windows права на выполнение не требуются")
        return
    
    print("🔧 Установка прав на выполнение для Unix-систем...")
    
    scripts_dir = Path(__file__).parent
    
    # Делаем все .py файлы исполняемыми
    for script in scripts_dir.glob('*.py'):
        if script.name != 'make_executable.py':  # себя не трогаем
            st = os.stat(script)
            os.chmod(script, st.st_mode | stat.S_IEXEC)
            print(f"  + {script.name}")
    
    print("✅ Готово")


if __name__ == "__main__":
    make_executable()