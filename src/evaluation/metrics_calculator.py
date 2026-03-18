"""
Расчет метрик для моделей
Путь: src/evaluation/metrics_calculator.py
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))


class MetricsCalculator:
    """Класс для расчета метрик классификации"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_metrics(self, y_true, y_pred, y_prob=None, model_name='model'):
        """
        Расчет всех метрик
        
        Args:
            y_true: истинные метки
            y_pred: предсказанные метки
            y_prob: вероятности (для ROC-AUC)
            model_name: название модели
        
        Returns:
            dict: словарь с метриками
        """
        metrics = {
            'model': model_name,
            'accuracy': accuracy_score(y_true, y_pred)
        }
        
        # Precision, Recall, F1 для каждого класса
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        
        # Для каждого класса отдельно
        classes = np.unique(y_true)
        for cls in classes:
            cls_name = 'human' if cls == 0 else 'robot'
            metrics[f'precision_{cls_name}'] = precision_score(y_true, y_pred, labels=[cls], average=None)[0]
            metrics[f'recall_{cls_name}'] = recall_score(y_true, y_pred, labels=[cls], average=None)[0]
            metrics[f'f1_{cls_name}'] = f1_score(y_true, y_pred, labels=[cls], average=None)[0]
        
        # ROC-AUC если есть вероятности
        if y_prob is not None:
            if len(np.unique(y_true)) == 2:  # бинарная классификация
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        
        # Матрица ошибок
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Дополнительные метрики
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        metrics['true_negative'] = int(tn)
        metrics['false_positive'] = int(fp)
        metrics['false_negative'] = int(fn)
        metrics['true_positive'] = int(tp)
        
        # Специфичность и чувствительность
        if (tn + fp) > 0:
            metrics['specificity'] = tn / (tn + fp)
        else:
            metrics['specificity'] = 0
        
        if (tp + fn) > 0:
            metrics['sensitivity'] = tp / (tp + fn)  # то же что recall для positive класса
        else:
            metrics['sensitivity'] = 0
        
        self.metrics = metrics
        return metrics
    
    def print_report(self, y_true, y_pred, target_names=None):
        """Печать отчета классификации"""
        if target_names is None:
            target_names = ['human', 'robot']
        
        report = classification_report(y_true, y_pred, target_names=target_names)
        print(report)
        
        return report
    
    def save_metrics(self, filepath):
        """Сохранение метрик в JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    @staticmethod
    def load_metrics(filepath):
        """Загрузка метрик из JSON"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def compare_metrics(metrics_list):
        """Сравнение нескольких наборов метрик"""
        import pandas as pd
        
        df = pd.DataFrame(metrics_list)
        return df