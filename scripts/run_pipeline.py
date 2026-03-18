#!/usr/bin/env python3
"""
Полный пайплайн обработки и обучения
"""

import subprocess
import sys
import time
from pathlib import Path

steps = [
    ("Аугментация (шум + телефон)", "src/data_preparation/03_augment_audio.py", ["--sr", "8000", "--force"]),
    ("Разбиение данных", "src/data_preparation/04_create_splits.py", []),
    ("Извлечение признаков", "src/features/build_feature_pipeline.py", ["--force"]),
    ("Обучение всех моделей", "src/training/train_all_models.py", ["--features", "acoustic", "--force"]),
]

for desc, script, args in steps:
    print(f"\n{'='*60}")
    print(f"🚀 {desc}")
    print(f"{'='*60}")
    
    cmd = [sys.executable, script] + args
    print(f"Выполняем: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"❌ Ошибка на этапе: {desc}")
        sys.exit(1)
    
    time.sleep(2)

print("\n" + "="*60)
print("✅ ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН")
print("="*60)