"""
Построение графика скорость-точность
Путь: src/evaluation/plot_speed_accuracy.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_speed_accuracy_tradeoff(metrics_df, plots_dir):
    """
    Построение графика компромисса между скоростью и точностью
    
    Args:
        metrics_df: DataFrame с метриками (должен содержать model, accuracy, inference_time)
        plots_dir: директория для сохранения
    """
    if 'inference_time' not in metrics_df.columns:
        print("Нет данных о времени инференса")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Определяем размер пузырьков по F1 или другому параметру
    if 'f1_weighted' in metrics_df.columns:
        sizes = metrics_df['f1_weighted'] * 1000
    else:
        sizes = np.ones(len(metrics_df)) * 100
    
    # Цвета по типу модели
    colors = []
    for model in metrics_df['Model']:
        if any(x in model.lower() for x in ['forest', 'xgboost', 'catboost', 'logistic']):
            colors.append('blue')  # традиционные ML
        elif any(x in model.lower() for x in ['cnn', 'lstm', 'hybrid']):
            colors.append('red')   # нейросети
        elif any(x in model.lower() for x in ['voting', 'stacking']):
            colors.append('green') # ансамбли
        else:
            colors.append('purple')
    
    scatter = plt.scatter(
        metrics_df['inference_time'] * 1000,  # в миллисекундах
        metrics_df['accuracy'],
        s=sizes,
        c=colors,
        alpha=0.6,
        edgecolors='black',
        linewidth=1
    )
    
    # Добавляем подписи
    for i, row in metrics_df.iterrows():
        plt.annotate(
            row['Model'],
            (row['inference_time'] * 1000, row['accuracy']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.8
        )
    
    plt.xlabel('Inference Time (ms)')
    plt.ylabel('Accuracy')
    plt.title('Speed-Accuracy Trade-off')
    plt.grid(True, alpha=0.3)
    
    # Легенда для типов моделей
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.6, label='Traditional ML'),
        Patch(facecolor='red', alpha=0.6, label='Deep Learning'),
        Patch(facecolor='green', alpha=0.6, label='Ensemble')
    ]
    plt.legend(handles=legend_elements)
    
    # Оптимальная область (верхний левый угол)
    plt.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=10, color='gray', linestyle='--', alpha=0.5)
    plt.text(1, 0.91, 'Target Region', fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'speed_accuracy_tradeoff.png', dpi=150)
    plt.close()
    
    print(f"✅ График скорость-точность сохранен в {plots_dir}")
    
    return plt.gcf()