#!/usr/bin/env python3
"""
Скрипт для сбора и сохранения подробной таблицы результатов всех моделей
Включает метрики: precision, recall, accuracy, f1-score, RAM, VRAM, устройство
Путь: src/evaluation/save_results_table.py
"""

import argparse
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import torch
import psutil
import GPUtil
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.file_utils import FileManager
from src.utils.logger import setup_logger
from src.utils.config_loader import ConfigLoader


class ResultsTableSaver:
    """Класс для сбора и сохранения таблицы результатов всех моделей"""
    
    def __init__(self, config_path='configs'):
        self.config_loader = ConfigLoader(config_path)
        self.file_manager = FileManager()
        
        # Загружаем пути
        try:
            paths_config = self.config_loader.load_config('paths_config')
            self.metrics_dir = Path(paths_config['paths']['metrics_root'])
            self.models_root = Path(paths_config['paths']['models_root'])
            self.results_dir = self.metrics_dir.parent / 'comparison'
        except:
            self.metrics_dir = Path('results/metrics')
            self.models_root = Path('results/trained_models')
            self.results_dir = Path('results/comparison')
        
        self.file_manager.ensure_dir(self.results_dir)
        self.logger = setup_logger('results_table')
        
        # Список всех моделей для сбора данных
        self.all_models = [
            # Нейросетевые модели
            {'name': 'CNN', 'file': 'cnn_metrics.json', 'type': 'neural', 'folder': 'cnn_gpu'},
            {'name': 'LSTM', 'file': 'lstm_metrics.json', 'type': 'neural', 'folder': 'lstm_gpu'},
            {'name': 'Hybrid CNN-LSTM', 'file': 'hybrid_metrics.json', 'type': 'neural', 'folder': 'hybrid_gpu'},
            
            # Традиционные модели
            {'name': 'Logistic Regression', 'file': 'logistic_metrics.json', 'type': 'traditional', 'folder': 'logistic'},
            {'name': 'Random Forest', 'file': 'random_forest_metrics.json', 'type': 'traditional', 'folder': 'random_forest'},
            {'name': 'XGBoost', 'file': 'xgboost_metrics.json', 'type': 'traditional', 'folder': 'xgboost'},
            {'name': 'CatBoost', 'file': 'catboost_metrics.json', 'type': 'traditional', 'folder': 'catboost'},
        ]
    
    def get_model_metrics(self, model_info):
        """Получение метрик из JSON файла модели"""
        metrics_file = self.metrics_dir / model_info['file']
        
        if not metrics_file.exists():
            self.logger.warning(f"Файл метрик не найден: {metrics_file}")
            return None
        
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            return metrics
        except Exception as e:
            self.logger.error(f"Ошибка загрузки {metrics_file}: {e}")
            return None
    
    def get_model_size(self, model_info):
        """Получение размера модели на диске"""
        if model_info['type'] == 'neural':
            model_file = self.models_root / model_info['folder'] / f"best_{model_info['folder'].replace('_gpu', '')}.pth"
        else:
            model_file = self.models_root / model_info['folder'] / 'model.pkl'
        
        if model_file.exists():
            size_bytes = model_file.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            return size_mb
        return 0
    
    def get_inference_time(self, model_info):
        """Получение времени инференса из метрик (если есть) или возврат None"""
        metrics = self.get_model_metrics(model_info)
        if metrics and 'inference_time' in metrics:
            return metrics['inference_time']
        return None
    
    def get_training_time(self, model_info):
        """Получение времени обучения из логов (приблизительно)"""
        # Можно вычислить из логов, но пока возвращаем None
        return None
    
    def get_device_info(self, model_info):
        """Получение информации об устройстве обучения"""
        metrics = self.get_model_metrics(model_info)
        if metrics and 'device' in metrics:
            return metrics['device']
        
        # Для традиционных моделей обычно CPU
        if model_info['type'] == 'traditional':
            return 'CPU'
        
        # Проверяем наличие GPU
        if torch.cuda.is_available():
            return f"GPU: {torch.cuda.get_device_name(0)}"
        return 'CPU'
    
    def get_ram_usage(self, model_info):
        """Получение потребления RAM при обучении (из логов или оценка)"""
        # Можно собрать из логов, но пока возвращаем приблизительные значения
        if model_info['type'] == 'neural':
            # Нейросетевые модели потребляют больше RAM
            return '~2-4 GB'
        else:
            # Традиционные модели потребляют меньше
            return '~0.5-1 GB'
    
    def get_vram_usage(self, model_info):
        """Получение потребления видеопамяти при обучении"""
        if model_info['type'] == 'neural':
            metrics = self.get_model_metrics(model_info)
            if metrics and 'gpu_memory' in metrics:
                return f"{metrics['gpu_memory']:.1f} GB"
            return '~2-3 GB'
        return 'N/A (CPU)'
    
    def collect_all_metrics(self):
        """Сбор всех метрик в единую таблицу"""
        self.logger.info("=" * 60)
        self.logger.info("📊 СБОР РЕЗУЛЬТАТОВ МОДЕЛЕЙ")
        self.logger.info("=" * 60)
        
        results = []
        
        for model in self.all_models:
            self.logger.info(f"\nОбработка: {model['name']}")
            
            metrics = self.get_model_metrics(model)
            
            if metrics is None:
                self.logger.warning(f"  Пропуск - метрики не найдены")
                continue
            
            # Извлекаем основные метрики
            row = {
                'Модель': model['name'],
                'Тип': model['type'],
                'Accuracy': metrics.get('accuracy', 0),
                'Precision (Human)': metrics.get('precision_human', 0),
                'Recall (Human)': metrics.get('recall_human', 0),
                'F1 (Human)': metrics.get('f1_human', 0),
                'Precision (Robot)': metrics.get('precision_robot', 0),
                'Recall (Robot)': metrics.get('recall_robot', 0),
                'F1 (Robot)': metrics.get('f1_robot', 0),
                'F1 (Weighted)': metrics.get('f1_weighted', 0),
                'ROC-AUC': metrics.get('roc_auc', 0),
                'Размер модели (MB)': round(self.get_model_size(model), 2),
                'Устройство': self.get_device_info(model),
                'RAM (оценка)': self.get_ram_usage(model),
                'VRAM (оценка)': self.get_vram_usage(model),
            }
            
            # Добавляем confusion matrix если есть
            if 'confusion_matrix' in metrics:
                cm = metrics['confusion_matrix']
                if len(cm) == 2 and len(cm[0]) == 2:
                    row['TP'] = cm[1][1]  # True Positive (робот определен как робот)
                    row['TN'] = cm[0][0]  # True Negative (человек определен как человек)
                    row['FP'] = cm[0][1]  # False Positive (человек определен как робот)
                    row['FN'] = cm[1][0]  # False Negative (робот определен как человек)
            
            results.append(row)
            self.logger.info(f"  ✅ Данные собраны")
        
        return pd.DataFrame(results)
    
    def save_table(self, df):
        """Сохранение таблицы в различных форматах"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("💾 СОХРАНЕНИЕ ТАБЛИЦЫ")
        self.logger.info("=" * 60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Сохраняем как CSV
        csv_path = self.results_dir / f'models_comparison_{timestamp}.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        self.logger.info(f"✅ CSV сохранен: {csv_path}")
        
        # 2. Сохраняем как Excel
        excel_path = self.results_dir / f'models_comparison_{timestamp}.xlsx'
        df.to_excel(excel_path, index=False, sheet_name='Model Comparison')
        self.logger.info(f"✅ Excel сохранен: {excel_path}")
        
        # 3. Сохраняем как Markdown
        md_path = self.results_dir / f'models_comparison_{timestamp}.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Сравнение моделей классификации\n\n")
            f.write(f"*Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            f.write(df.to_markdown(index=False))
        self.logger.info(f"✅ Markdown сохранен: {md_path}")
        
        # 4. Сохраняем как LaTeX (для научных публикаций)
        latex_path = self.results_dir / f'models_comparison_{timestamp}.tex'
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Сравнение моделей классификации}\n")
            # Выбираем основные колонки для LaTeX
            cols_to_show = ['Модель', 'Accuracy', 'F1 (Human)', 'F1 (Robot)', 'Размер модели (MB)', 'Устройство']
            df_latex = df[cols_to_show]
            f.write(df_latex.to_latex(index=False, float_format="%.4f"))
            f.write("\\label{tab:models_comparison}\n")
            f.write("\\end{table}\n")
        self.logger.info(f"✅ LaTeX сохранен: {latex_path}")
        
        # 5. Сохраняем как JSON
        json_path = self.results_dir / f'models_comparison_{timestamp}.json'
        df.to_json(json_path, orient='records', force_ascii=False, indent=2)
        self.logger.info(f"✅ JSON сохранен: {json_path}")
        
        return csv_path, excel_path
    
    def print_summary(self, df):
        """Вывод сводной таблицы в консоль"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
        self.logger.info("=" * 60)
        
        # Выбираем основные колонки для отображения
        display_cols = ['Модель', 'Тип', 'Accuracy', 'F1 (Human)', 'F1 (Robot)', 'Размер модели (MB)', 'Устройство']
        
        # Сортируем по Accuracy
        df_sorted = df.sort_values('Accuracy', ascending=False)
        
        # Форматируем значения
        for col in ['Accuracy', 'F1 (Human)', 'F1 (Robot)']:
            if col in df_sorted.columns:
                df_sorted[col] = df_sorted[col].apply(lambda x: f"{x:.4f}")
        
        # Выводим таблицу
        self.logger.info("\n" + df_sorted[display_cols].to_string(index=False))
        
        # Находим лучшую модель
        best_accuracy = df.loc[df['Accuracy'].idxmax()]
        best_f1_robot = df.loc[df['F1 (Robot)'].idxmax()]
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("🏆 ЛУЧШИЕ МОДЕЛИ")
        self.logger.info("=" * 60)
        self.logger.info(f"По точности (Accuracy): {best_accuracy['Модель']} - {best_accuracy['Accuracy']:.4f}")
        self.logger.info(f"По F1 для роботов: {best_f1_robot['Модель']} - {best_f1_robot['F1 (Robot)']:.4f}")
    
    def run(self):
        """Запуск сбора и сохранения результатов"""
        self.logger.info("=" * 60)
        self.logger.info("📊 СБОР И СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        self.logger.info("=" * 60)
        
        # Собираем данные
        df = self.collect_all_metrics()
        
        if df.empty:
            self.logger.error("❌ Нет данных для сохранения")
            return
        
        # Сохраняем таблицу
        self.save_table(df)
        
        # Выводим сводку
        self.print_summary(df)
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("✅ ГОТОВО")
        self.logger.info(f"📁 Результаты сохранены в: {self.results_dir}")
        self.logger.info("=" * 60)
        
        return df


def main():
    parser = argparse.ArgumentParser(description='Сохранение таблицы результатов моделей')
    parser.add_argument('--config', type=str, default='configs', help='Путь к конфигам')
    parser.add_argument('--output', type=str, default=None, help='Путь для сохранения результатов')
    
    args = parser.parse_args()
    
    saver = ResultsTableSaver(config_path=args.config)
    
    if args.output:
        saver.results_dir = Path(args.output)
        saver.file_manager.ensure_dir(saver.results_dir)
    
    saver.run()


if __name__ == "__main__":
    main()