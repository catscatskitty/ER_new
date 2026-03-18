#!/usr/bin/env python3
"""
Скрипт для установки всех зависимостей
Путь: scripts/install_dependencies.py
"""

import subprocess
import sys
import platform
from pathlib import Path


class DependencyInstaller:
    """Класс для установки зависимостей"""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent
        self.python_cmd = 'python' if platform.system() == 'Windows' else 'python3'
        
        # Все необходимые пакеты
        self.packages = [
            'pyyaml>=5.4.0',
            'numpy>=1.21.0',
            'pandas>=1.3.0',
            'scipy>=1.7.0',
            'matplotlib>=3.4.0',
            'seaborn>=0.11.0',
            'scikit-learn>=1.0.0',
            'joblib>=1.0.0',
            'tqdm>=4.62.0',
            'librosa>=0.9.0',
            'soundfile>=0.10.0',
            'noisereduce>=2.0.0',
            'xgboost>=1.5.0',
            'catboost>=1.0.0',
            'tensorflow>=2.8.0',
            'transformers>=4.16.0',
            'streamlit>=1.10.0',
            'plotly>=5.6.0',
            'jupyter>=1.0.0',
            'ipykernel>=6.0.0',
            'pydub>=0.25.1',
            'audioread>=2.1.9'
        ]
    
    def install_pip(self):
        """Обновление pip"""
        print("📦 Обновление pip...")
        subprocess.run(
            [self.python_cmd, '-m', 'pip', 'install', '--upgrade', 'pip'],
            check=False
        )
    
    def install_packages(self):
        """Установка всех пакетов"""
        print("="*60)
        print("📦 УСТАНОВКА ВСЕХ ЗАВИСИМОСТЕЙ")
        print("="*60)
        
        # Обновляем pip
        self.install_pip()
        
        # Устанавливаем пакеты
        for package in self.packages:
            print(f"\n🔧 Установка {package}...")
            try:
                subprocess.run(
                    [self.python_cmd, '-m', 'pip', 'install', package],
                    check=True
                )
                print(f"  ✅ {package} установлен")
            except subprocess.CalledProcessError as e:
                print(f"  ❌ Ошибка при установке {package}: {e}")
        
        print("\n" + "="*60)
        print("✅ УСТАНОВКА ЗАВЕРШЕНА")
        print("="*60)
        
        # Проверка
        self.check_installation()
    
    def check_installation(self):
        """Проверка установки"""
        print("\n🔍 Проверка установки...")
        
        packages_to_check = [
            ('yaml', 'pyyaml'),
            ('numpy', 'numpy'),
            ('pandas', 'pandas'),
            ('librosa', 'librosa'),
            ('sklearn', 'scikit-learn'),
            ('tensorflow', 'tensorflow'),
            ('streamlit', 'streamlit')
        ]
        
        all_ok = True
        for import_name, package_name in packages_to_check:
            try:
                __import__(import_name)
                print(f"  ✅ {package_name}")
            except ImportError:
                print(f"  ❌ {package_name} (не удалось импортировать {import_name})")
                all_ok = False
        
        if all_ok:
            print("\n✅ Все пакеты успешно установлены!")
        else:
            print("\n⚠️ Некоторые пакеты не установились. Попробуйте установить вручную:")
            print(f"  {self.python_cmd} -m pip install -r requirements.txt")
    
    def create_requirements(self):
        """Создание requirements.txt"""
        requirements_path = self.root_dir / 'requirements.txt'
        
        with open(requirements_path, 'w') as f:
            for package in self.packages:
                f.write(f"{package}\n")
        
        print(f"✅ requirements.txt создан в {requirements_path}")


if __name__ == "__main__":
    installer = DependencyInstaller()
    
    # Создаем requirements.txt
    installer.create_requirements()
    
    # Спрашиваем, устанавливать ли
    response = input("\nУстановить все зависимости? (y/n): ").strip().lower()
    if response == 'y':
        installer.install_packages()
    else:
        print("\nДля установки позже выполните:")
        print(f"  {installer.python_cmd} -m pip install -r requirements.txt")