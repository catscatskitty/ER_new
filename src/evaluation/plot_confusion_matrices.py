import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrices(results_dict, save_path='results/plots/confusion_matrices.png'):
    """Построение матриц ошибок для всех моделей"""
    n_models = len(results_dict)
    fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (model_name, metrics) in enumerate(results_dict.items()):
        if 'confusion_matrix' not in metrics:
            continue
        
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(model_name)
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()