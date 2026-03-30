import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path


def plot_roc_curves(results_dict, save_path='results/plots/roc_curves.png'):
    """Построение ROC-кривых для всех моделей"""
    plt.figure(figsize=(10, 8))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A4E9B', '#00A896']
    colors = colors * (len(results_dict) // len(colors) + 1)
    
    for idx, (model_name, metrics) in enumerate(results_dict.items()):
        if 'fpr' in metrics and 'tpr' in metrics:
            fpr = metrics['fpr']
            tpr = metrics['tpr']
            roc_auc = metrics.get('auc', auc(fpr, tpr))
            
            plt.plot(fpr, tpr, color=colors[idx % len(colors)],
                     lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC curves saved to {save_path}")