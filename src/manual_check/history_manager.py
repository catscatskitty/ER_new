"""
Управление историей проверок
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import json


class HistoryManager:
    def __init__(self, history_dir):
        self.history_dir = Path(history_dir)
        try:
            self.history_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"⚠️ Ошибка создания директории: {e}")
        
        self.history_file = self.history_dir / 'check_history.csv'
        self.history = self._load_history()
    
    def _load_history(self):
        if self.history_file.exists():
            try:
                df = pd.read_csv(self.history_file)
                return df.to_dict('records')
            except Exception as e:
                print(f"⚠️ Ошибка загрузки истории: {e}")
                return []
        return []
    
    def _save_history(self):
        try:
            if self.history:
                df = pd.DataFrame(self.history)
                df.to_csv(self.history_file, index=False, encoding='utf-8')
        except Exception as e:
            print(f"⚠️ Ошибка сохранения истории: {e}")
    
    def add_entry(self, filename, results):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'final_prediction': results['final_prediction'],
            'average_confidence': results['average_confidence'],
            'human_votes': results['human_votes'],
            'robot_votes': results['robot_votes'],
            'total_models': len(results['model_predictions'])
        }
        entry['details'] = json.dumps(results['model_predictions'], ensure_ascii=False)
        self.history.append(entry)
        if len(self.history) > 100:
            self.history = self.history[-100:]
        self._save_history()
    
    def get_history(self):
        return self.history
    
    def get_recent(self, n=10):
        return self.history[-n:] if self.history else []
    
    def clear_history(self):
        self.history = []
        if self.history_file.exists():
            try:
                self.history_file.unlink()
            except:
                pass
    
    def export_history(self, format='csv'):
        if format == 'csv':
            df = pd.DataFrame(self.history)
            export_path = self.history_dir / f'history_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            df.to_csv(export_path, index=False, encoding='utf-8')
            return export_path
        elif format == 'json':
            export_path = self.history_dir / f'history_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
            return export_path
    
    def get_statistics(self):
        if not self.history:
            return {}
        df = pd.DataFrame(self.history)
        return {
            'total_checks': len(df),
            'human_count': int((df['final_prediction'] == 'human').sum()),
            'robot_count': int((df['final_prediction'] == 'robot').sum()),
            'avg_confidence': float(df['average_confidence'].mean()),
            'avg_human_votes': float(df['human_votes'].mean()),
            'avg_robot_votes': float(df['robot_votes'].mean())
        }