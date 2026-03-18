#!/usr/bin/env python
"""
Только оценка моделей
"""

import os
import sys
import subprocess
from pathlib import Path

def evaluate_only():
    """
    Запуск только оценки моделей
    """
    print("=" * 60)
    print("ЗАПУСК ОЦЕНКИ МОДЕЛЕЙ")
    print("=" * 60)
    
    # Проверка наличия метрик
    metrics_dir = Path('results/metrics')
    if not metrics_dir.exists():
        print("❌ Директория с метриками не найдена")
        return False
    
    # Запуск скриптов оценки
    scripts = [
        'src/evaluation/compare_models.py',
        'src/evaluation/plot_confusion_matrices.py',
        'src/evaluation/plot_roc_curves.py',
        'src/evaluation/plot_feature_importance.py',
        'src/evaluation/generate_comparison_table.py'
    ]
    
    for script in scripts:
        script_path = Path(__file__).parent.parent / script
        if not script_path.exists():
            print(f"❌ Скрипт не найден: {script}")
            continue
        
        print(f"\n--- Запуск {script} ---")
        try:
            subprocess.run([sys.executable, str(script_path)])
        except Exception as e:
            print(f"Ошибка при запуске {script}: {e}")
    
    print("\n✅ Оценка завершена")
    print(f"Результаты сохранены в: {metrics_dir}")
    return True

if __name__ == "__main__":
    evaluate_only()