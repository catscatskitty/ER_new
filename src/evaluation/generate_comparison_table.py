import pandas as pd
import json
from pathlib import Path
import logging
from tabulate import tabulate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_comparison_table():
    """
    Генерация таблицы сравнения всех моделей
    """
    metrics_dir = Path('results/metrics')
    
    all_metrics = []
    
    # Сбор всех метрик
    for metrics_file in metrics_dir.glob('*.json'):
        if metrics_file.name == 'all_models_comparison.csv':
            continue
            
        with open(metrics_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                
                # Обработка разных форматов
                if 'model' in data:
                    all_metrics.append(data)
                elif isinstance(data, dict):
                    for model_name, model_data in data.items():
                        if isinstance(model_data, dict):
                            model_data['model'] = model_name
                            all_metrics.append(model_data)
            except:
                continue
    
    if not all_metrics:
        logger.warning("Нет данных для сравнения")
        return
    
    # Создание DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Выбор нужных колонок
    columns_to_keep = ['model', 'test_accuracy', 'test_f1', 'test_auc', 'train_accuracy', 'val_accuracy']
    available_columns = [col for col in columns_to_keep if col in df.columns]
    
    df_display = df[available_columns].copy()
    
    # Сортировка по accuracy
    if 'test_accuracy' in df_display.columns:
        df_display = df_display.sort_values('test_accuracy', ascending=False)
    
    # Форматирование
    for col in ['test_accuracy', 'test_f1', 'test_auc', 'train_accuracy', 'val_accuracy']:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    
    # Сохранение в CSV
    output_file = metrics_dir / 'all_models_comparison.csv'
    df_display.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    # Сохранение в Markdown
    markdown_table = tabulate(df_display, headers='keys', tablefmt='pipe', showindex=False)
    
    with open(metrics_dir / 'comparison_table.md', 'w', encoding='utf-8') as f:
        f.write("# Сравнение моделей\n\n")
        f.write(markdown_table)
    
    # Вывод в консоль
    logger.info("\n" + "="*80)
    logger.info("ТАБЛИЦА СРАВНЕНИЯ МОДЕЛЕЙ")
    logger.info("="*80)
    print("\n" + markdown_table)
    
    # Статистика
    logger.info("\n" + "="*80)
    logger.info("СТАТИСТИКА")
    logger.info("="*80)
    
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if 'test_accuracy' in numeric_df.columns:
        logger.info(f"Лучшая модель по accuracy: {df.loc[df['test_accuracy'].idxmax(), 'model']} "
                   f"({df['test_accuracy'].max():.4f})")
        logger.info(f"Худшая модель по accuracy: {df.loc[df['test_accuracy'].idxmin(), 'model']} "
                   f"({df['test_accuracy'].min():.4f})")
        logger.info(f"Средняя accuracy: {df['test_accuracy'].mean():.4f}")
        logger.info(f"Медианная accuracy: {df['test_accuracy'].median():.4f}")
    
    if 'test_f1' in numeric_df.columns:
        logger.info(f"\nЛучшая модель по F1: {df.loc[df['test_f1'].idxmax(), 'model']} "
                   f"({df['test_f1'].max():.4f})")
    
    if 'test_auc' in numeric_df.columns:
        logger.info(f"\nЛучшая модель по AUC: {df.loc[df['test_auc'].idxmax(), 'model']} "
                   f"({df['test_auc'].max():.4f})")
    
    return df_display

if __name__ == "__main__":
    generate_comparison_table()