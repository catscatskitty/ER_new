import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_feature_importance():
    """
    Построение графиков важности признаков
    """
    metrics_dir = Path('results/metrics')
    plots_dir = Path('results/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Загрузка важности признаков из Random Forest
    rf_metrics_file = metrics_dir / 'random_forest_metrics.json'
    
    if not rf_metrics_file.exists():
        logger.warning("Нет данных о важности признаков")
        return
    
    with open(rf_metrics_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'feature_importance' not in data:
        logger.warning("Нет данных о важности признаков в файле")
        return
    
    importance = np.array(data['feature_importance'])
    
    # Загрузка названий признаков
    feature_names_file = Path('data/processed/feature_names.txt')
    if feature_names_file.exists():
        with open(feature_names_file, 'r', encoding='utf-8') as f:
            feature_names = [line.strip() for line in f.readlines()]
    else:
        feature_names = [f'Feature_{i}' for i in range(len(importance))]
    
    # Сортировка по важности
    sorted_idx = np.argsort(importance)[::-1]
    
    # Топ-20 признаков
    top_n = min(20, len(importance))
    
    plt.figure(figsize=(12, 8))
    
    plt.barh(range(top_n), importance[sorted_idx[:top_n]][::-1])
    plt.yticks(range(top_n), [feature_names[i] for i in sorted_idx[:top_n]][::-1])
    
    plt.xlabel('Важность')
    plt.title('Топ-20 наиболее важных признаков (Random Forest)')
    plt.tight_layout()
    
    plt.savefig(plots_dir / 'feature_importance_top20.png', dpi=150, bbox_inches='tight')
    plt.savefig(plots_dir / 'feature_importance_top20.pdf', bbox_inches='tight')
    
    # Все признаки
    plt.figure(figsize=(14, 10))
    
    plt.barh(range(len(importance)), importance[sorted_idx])
    plt.yticks(range(len(importance)), [feature_names[i] for i in sorted_idx], fontsize=6)
    
    plt.xlabel('Важность')
    plt.title('Важность всех признаков (Random Forest)')
    plt.tight_layout()
    
    plt.savefig(plots_dir / 'feature_importance_all.png', dpi=150, bbox_inches='tight')
    
    logger.info(f"Графики важности признаков сохранены в {plots_dir}")

if __name__ == "__main__":
    plot_feature_importance()