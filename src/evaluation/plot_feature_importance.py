import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_feature_importance(importance_dict, feature_names, 
                            save_path='results/plots/feature_importance.png', 
                            top_n=20):
    """Построение графика важности признаков"""
    fig, axes = plt.subplots(len(importance_dict), 1, 
                              figsize=(12, 4 * len(importance_dict)))
    
    if len(importance_dict) == 1:
        axes = [axes]
    
    for idx, (model_name, importance) in enumerate(importance_dict.items()):
        if importance is None:
            axes[idx].text(0.5, 0.5, f'{model_name}: no feature importance available', 
                          ha='center', va='center')
            axes[idx].set_title(model_name)
            continue
        
        # Сортировка
        sorted_idx = np.argsort(importance)[::-1][:top_n]
        sorted_importance = importance[sorted_idx]
        sorted_names = [feature_names[i] for i in sorted_idx]
        
        axes[idx].barh(range(len(sorted_importance)), sorted_importance, color='#2E86AB')
        axes[idx].set_yticks(range(len(sorted_importance)))
        axes[idx].set_yticklabels(sorted_names)
        axes[idx].set_xlabel('Importance')
        axes[idx].set_title(f'{model_name} - Top {top_n} Features')
        axes[idx].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Feature importance saved to {save_path}")