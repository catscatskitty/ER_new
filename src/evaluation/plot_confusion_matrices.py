import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_confusion_matrices():
    """
    Построение матриц ошибок для всех моделей
    """
    metrics_dir = Path('results/metrics')
    plots_dir = Path('results/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Сбор всех метрик с матрицами ошибок
    confusion_matrices = {}
    
    for metrics_file in metrics_dir.glob('*.json'):
        with open(metrics_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                
                # Обработка разных форматов
                if 'confusion_matrix' in data:
                    confusion_matrices[data.get('model', metrics_file.stem)] = data['confusion_matrix']
                elif isinstance(data, dict):
                    for model_name, model_data in data.items():
                        if isinstance(model_data, dict) and 'confusion_matrix' in model_data:
                            confusion_matrices[model_name] = model_data['confusion_matrix']
            except:
                continue
    
    if not confusion_matrices:
        logger.warning("Нет данных для построения матриц ошибок")
        return
    
    # Определяем размер сетки
    n_models = len(confusion_matrices)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (model_name, cm) in enumerate(confusion_matrices.items()):
        ax = axes[idx]
        
        # Конвертируем в numpy если нужно
        if isinstance(cm, list):
            cm = np.array(cm)
        
        # Нормализация
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Построение
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=['Человек', 'Робот'],
                   yticklabels=['Человек', 'Робот'], ax=ax)
        
        ax.set_title(f'{model_name}\nAccuracy: {np.trace(cm)/np.sum(cm):.3f}')
        ax.set_xlabel('Предсказано')
        ax.set_ylabel('Истинное значение')
    
    # Скрываем лишние подграфики
    for idx in range(len(confusion_matrices), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.savefig(plots_dir / 'confusion_matrices.pdf', bbox_inches='tight')
    
    logger.info(f"Матрицы ошибок сохранены в {plots_dir}")

if __name__ == "__main__":
    plot_confusion_matrices()