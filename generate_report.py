import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Настройки
METRICS_DIR = Path('results/metrics')
PLOTS_DIR = Path('results/plots/final_report')
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def load_metrics_from_json(filepath, feature_type):
    """Извлечь метрики из JSON-файла"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    model_name = filepath.stem.replace('_metrics', '').replace('_combined', '')
    
    if '_combined' in filepath.stem:
        ft = 'combined'
    else:
        ft = feature_type
    
    metrics = {
        'Model': model_name,
        'FeatureType': ft,
        'Accuracy': data.get('accuracy', None),
        'F1_Weighted': data.get('f1_score', data.get('f1_weighted', None)),
        'F1_Human': None,
        'F1_Robot': None,
        'Precision_Human': None,
        'Recall_Human': None,
        'Precision_Robot': None,
        'Recall_Robot': None,
    }
    
    if 'classification_report' in data:
        report = data['classification_report']
        if isinstance(report, dict):
            for label, vals in report.items():
                if label in ['0', 'human', 'Human']:
                    metrics['F1_Human'] = vals.get('f1-score', None)
                    metrics['Precision_Human'] = vals.get('precision', None)
                    metrics['Recall_Human'] = vals.get('recall', None)
                elif label in ['1', 'robot', 'Robot']:
                    metrics['F1_Robot'] = vals.get('f1-score', None)
                    metrics['Precision_Robot'] = vals.get('precision', None)
                    metrics['Recall_Robot'] = vals.get('recall', None)
    
    return metrics

def load_metrics_from_csv(filepath):
    """Загрузить метрики из CSV-файла"""
    df = pd.read_csv(filepath)
    # Приводим колонки к единому формату, если нужно
    # Предполагаем, что там есть Model, FeatureType, Accuracy, F1_Weighted и т.д.
    return df

def main():
    all_metrics = []
    
    # 1. Загружаем из all_models_metrics.csv, если он есть
    csv_path = METRICS_DIR / 'all_models_metrics.csv'
    if csv_path.exists():
        df_csv = load_metrics_from_csv(csv_path)
        all_metrics.append(df_csv)
        print(f"Loaded {len(df_csv)} entries from {csv_path.name}")
    
    # 2. Загружаем из JSON-файлов
    for json_path in METRICS_DIR.glob('*.json'):
        if json_path.stem in ['best_models', 'training_report', 'all_models_comparison']:
            continue
        
        if '_combined' in json_path.stem:
            ft = 'combined'
        else:
            ft = 'acoustic'
        
        try:
            metrics = load_metrics_from_json(json_path, ft)
            all_metrics.append(pd.DataFrame([metrics]))
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
    
    if not all_metrics:
        print("No metrics found. Exiting.")
        return
    
    df = pd.concat(all_metrics, ignore_index=True)
    
    # Удаляем строки с пропущенными значениями в ключевых колонках
    df = df.dropna(subset=['Model', 'Accuracy', 'F1_Weighted']).copy()
    
    # Удаляем дубликаты (одинаковые модель+тип признаков)
    df = df.drop_duplicates(subset=['Model', 'FeatureType'], keep='first')
    
    # Сохраняем сводную таблицу
    df.to_csv(PLOTS_DIR / 'summary_metrics.csv', index=False)
    print(f"Saved summary metrics to {PLOTS_DIR / 'summary_metrics.csv'}")
    
    # 3. Визуализация
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df, x='Model', y='Accuracy', hue='FeatureType')
    plt.title('Model Accuracy by Feature Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'accuracy_comparison.png', dpi=150)
    plt.close()
    
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df, x='Model', y='F1_Weighted', hue='FeatureType')
    plt.title('Weighted F1 Score by Feature Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'f1_comparison.png', dpi=150)
    plt.close()
    
    # 4. Лучшие модели
    best_models = df.groupby('Model').apply(lambda x: x.loc[x['Accuracy'].idxmax()]).reset_index(drop=True)
    best_models = best_models.sort_values('Accuracy', ascending=False).head(5)
    print("\nTop 5 models overall:")
    print(best_models[['Model', 'FeatureType', 'Accuracy', 'F1_Weighted']])
    
    # 5. Формирование Markdown-отчёта
    report_lines = []
    report_lines.append("# Итоговый отчёт по проекту «Человек vs Робот»\n")
    report_lines.append("## Сравнение моделей на акустических, фонетических и комбинированных признаках\n")
    
    report_lines.append("### Сводная таблица метрик\n")
    report_lines.append(df.to_markdown(index=False))
    
    report_lines.append("\n### Лучшие модели по Accuracy\n")
    report_lines.append(best_models[['Model', 'FeatureType', 'Accuracy', 'F1_Weighted']].to_markdown(index=False))
    
    report_lines.append("\n### Графики\n")
    report_lines.append("![Accuracy comparison](accuracy_comparison.png)")
    report_lines.append("![F1 comparison](f1_comparison.png)")
    
    report_lines.append("\n### Выводы\n")
    # Автоматические выводы
    best_acoustic = df[df['FeatureType']=='acoustic'].sort_values('Accuracy', ascending=False).iloc[0] if 'acoustic' in df['FeatureType'].values else None
    best_phonetic = df[df['FeatureType']=='phonetic'].sort_values('Accuracy', ascending=False).iloc[0] if 'phonetic' in df['FeatureType'].values else None
    best_combined = df[df['FeatureType']=='combined'].sort_values('Accuracy', ascending=False).iloc[0] if 'combined' in df['FeatureType'].values else None
    
    if best_acoustic is not None:
        report_lines.append(f"- **Лучшая модель на акустических признаках**: {best_acoustic['Model']} с Accuracy = {best_acoustic['Accuracy']:.4f}")
    if best_phonetic is not None:
        report_lines.append(f"- **Лучшая модель на фонетических признаках**: {best_phonetic['Model']} с Accuracy = {best_phonetic['Accuracy']:.4f}")
    if best_combined is not None:
        report_lines.append(f"- **Лучшая модель на комбинированных признаках**: {best_combined['Model']} с Accuracy = {best_combined['Accuracy']:.4f}")
    
    if 'acoustic' in df['FeatureType'].values and 'combined' in df['FeatureType'].values:
        avg_acoustic = df[df['FeatureType']=='acoustic']['Accuracy'].mean()
        avg_combined = df[df['FeatureType']=='combined']['Accuracy'].mean()
        report_lines.append(f"- **Средняя точность по всем моделям**: акустические {avg_acoustic:.4f}, комбинированные {avg_combined:.4f}")
    
    report_lines.append("\n### Заключение\n")
    report_lines.append("На основе полученных результатов... (добавьте ручные выводы)")
    
    with open(PLOTS_DIR / 'final_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Report saved to {PLOTS_DIR / 'final_report.md'}")
    print("Done!")

if __name__ == "__main__":
    main()