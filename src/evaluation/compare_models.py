#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import logging
from tabulate import tabulate

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_all_models():
    """
    Сравнение всех обученных моделей с разбивкой по типам признаков
    """
    metrics_dir = project_root / 'results' / 'metrics'
    
    all_metrics = []
    
    # Собираем все метрики
    for metrics_file in metrics_dir.glob('*.json'):
        if 'comparison' in metrics_file.name:
            continue
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Определяем тип признаков из имени файла или поля
                if 'feature_type' in data:
                    feature_type = data['feature_type']
                else:
                    if 'acoustic' in metrics_file.name:
                        feature_type = 'acoustic'
                    elif 'phonetic' in metrics_file.name:
                        feature_type = 'phonetic'
                    elif 'combined' in metrics_file.name:
                        feature_type = 'combined'
                    else:
                        feature_type = 'unknown'
                
                all_metrics.append({
                    'model': data.get('model', metrics_file.stem),
                    'feature_type': feature_type,
                    'test_accuracy': data.get('test_accuracy', 0),
                    'test_f1': data.get('test_f1', 0),
                    'test_auc': data.get('test_auc', 0),
                    'feature_count': data.get('feature_count', 0)
                })
        except Exception as e:
            logger.warning(f"Ошибка загрузки {metrics_file}: {e}")
    
    if not all_metrics:
        logger.error("Нет метрик для сравнения")
        return
    
    # Создаем DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Сортируем
    df = df.sort_values(['feature_type', 'test_accuracy'], ascending=[True, False])
    
    # Сохраняем полную таблицу
    full_csv = metrics_dir / 'all_models_comparison.csv'
    df.to_csv(full_csv, index=False, encoding='utf-8-sig')
    
    # Сводная таблица по типам признаков
    pivot = df.pivot_table(
        values=['test_accuracy', 'test_f1', 'test_auc'],
        index='model',
        columns='feature_type',
        aggfunc='first'
    )
    
    # Форматируем для вывода
    display_df = df.copy()
    for col in ['test_accuracy', 'test_f1', 'test_auc']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    logger.info("\n" + "="*100)
    logger.info("СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ ПО ТИПАМ ПРИЗНАКОВ")
    logger.info("="*100)
    
    # Группировка по feature_type
    for ftype in df['feature_type'].unique():
        ftype_df = df[df['feature_type'] == ftype].copy()
        ftype_df = ftype_df.sort_values('test_accuracy', ascending=False)
        
        logger.info(f"\n--- {ftype.upper()} признаки ({len(ftype_df)} моделей) ---")
        table = ftype_df[['model', 'test_accuracy', 'test_f1', 'test_auc']].head(10)
        logger.info(f"\n{tabulate(table, headers='keys', tablefmt='grid', showindex=False)}")
    
    # Статистика
    logger.info("\n" + "="*100)
    logger.info("СТАТИСТИКА ПО ТИПАМ ПРИЗНАКОВ")
    logger.info("="*100)
    
    stats = df.groupby('feature_type').agg({
        'test_accuracy': ['mean', 'std', 'max', 'min'],
        'test_f1': ['mean'],
        'test_auc': ['mean']
    }).round(4)
    
    logger.info(f"\n{tabulate(stats, headers='keys', tablefmt='grid')}")
    
    # Лучшие модели
    logger.info("\n" + "="*100)
    logger.info("ЛУЧШИЕ МОДЕЛИ")
    logger.info("="*100)
    
    for ftype in df['feature_type'].unique():
        best = df[df['feature_type'] == ftype].sort_values('test_accuracy', ascending=False).iloc[0]
        logger.info(f"\n{ftype}: {best['model']} - {best['test_accuracy']:.4f}")
    
    # Сравнение acoustic vs combined
    if 'acoustic' in df['feature_type'].values and 'combined' in df['feature_type'].values:
        acoustic_mean = df[df['feature_type'] == 'acoustic']['test_accuracy'].mean()
        combined_mean = df[df['feature_type'] == 'combined']['test_accuracy'].mean()
        improvement = combined_mean - acoustic_mean
        
        logger.info("\n" + "="*100)
        logger.info("СРАВНЕНИЕ ACOUSTIC vs COMBINED")
        logger.info("="*100)
        logger.info(f"Acoustic средняя точность: {acoustic_mean:.4f}")
        logger.info(f"Combined средняя точность: {combined_mean:.4f}")
        logger.info(f"Улучшение: +{improvement:.4f} ({improvement/acoustic_mean*100:.1f}%)")
    
    return df

if __name__ == "__main__":
    compare_all_models()