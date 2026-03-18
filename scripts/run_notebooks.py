#!/usr/bin/env python3
"""
Универсальный скрипт для запуска Jupyter notebooks
Путь: scripts/run_notebooks.py
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger


class NotebookRunner:
    """Класс для запуска Jupyter notebooks"""
    
    def __init__(self):
        self.logger = setup_logger('notebook_runner')
        self.root_dir = Path(__file__).parent.parent
        
        # Определяем команду python
        import platform
        self.python_cmd = 'python' if platform.system() == 'Windows' else 'python3'
    
    def check_notebooks(self):
        """Проверка наличия notebooks"""
        notebooks_dir = self.root_dir / 'notebooks'
        if not notebooks_dir.exists():
            self.logger.error("❌ Директория notebooks не найдена!")
            return False
        
        notebooks = list(notebooks_dir.glob('*.ipynb'))
        if not notebooks:
            self.logger.warning("⚠️ Notebooks не найдены!")
            return False
        
        self.logger.info(f"✅ Найдено notebooks: {len(notebooks)}")
        for nb in notebooks:
            self.logger.info(f"  - {nb.name}")
        
        return True
    
    def run(self):
        """Запуск Jupyter"""
        self.logger.info("="*50)
        self.logger.info("📓 ЗАПУСК JUPYTER NOTEBOOKS")
        self.logger.info("="*50)
        
        # Проверяем наличие notebooks
        self.check_notebooks()
        
        # Проверяем наличие jupyter
        try:
            subprocess.run([self.python_cmd, '-m', 'jupyter', '--version'], 
                         capture_output=True, check=True)
        except:
            self.logger.error("❌ Jupyter не установлен!")
            self.logger.error("Установите: pip install jupyter")
            return
        
        self.logger.info("\n🚀 Запуск Jupyter...")
        self.logger.info("Интерфейс будет доступен по адресу: http://localhost:8888")
        self.logger.info("Для остановки нажмите Ctrl+C\n")
        
        # Открываем браузер
        def open_browser():
            time.sleep(3)
            webbrowser.open('http://localhost:8888/tree')
        
        import threading
        threading.Thread(target=open_browser, daemon=True).start()
        
        # Запускаем jupyter
        notebooks_dir = self.root_dir / 'notebooks'
        cmd = [self.python_cmd, '-m', 'jupyter', 'notebook', 
               '--notebook-dir', str(notebooks_dir)]
        
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            self.logger.info("\n\n👋 Jupyter остановлен")
        except Exception as e:
            self.logger.error(f"❌ Ошибка при запуске: {e}")


if __name__ == "__main__":
    runner = NotebookRunner()
    runner.run()