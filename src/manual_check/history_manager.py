import pandas as pd
from pathlib import Path
from datetime import datetime


class HistoryManager:
    """Управление историей проверок"""
    
    def __init__(self, history_dir='results/manual_checks'):
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.history_dir / 'check_history.csv'
        self._init_history()
    
    def _init_history(self):
        """Инициализация файла истории"""
        if not self.history_file.exists():
            df = pd.DataFrame(columns=[
                'timestamp', 'filename', 'final_prediction', 
                'average_confidence', 'human_votes', 'robot_votes',
                'model_predictions'
            ])
            df.to_csv(self.history_file, index=False)
    
    def add_entry(self, filename, results):
        """Добавление записи в историю"""
        df = pd.read_csv(self.history_file)
        
        new_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'filename': filename,
            'final_prediction': results['final_prediction'],
            'average_confidence': results['average_confidence'],
            'human_votes': results['human_votes'],
            'robot_votes': results['robot_votes'],
            'model_predictions': str(results['model_predictions'])
        }
        
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_csv(self.history_file, index=False)
    
    def get_history(self):
        """Получение всей истории"""
        try:
            df = pd.read_csv(self.history_file)
            return df.to_dict('records')
        except:
            return []
    
    def clear_history(self):
        """Очистка истории"""
        self._init_history()
        print("History cleared")
    
    def get_statistics(self):
        """Получение статистики по истории"""
        df = pd.read_csv(self.history_file)
        if len(df) == 0:
            return {'total': 0, 'human': 0, 'robot': 0, 'avg_confidence': 0}
        
        human_count = (df['final_prediction'] == 'human').sum()
        robot_count = (df['final_prediction'] == 'robot').sum()
        
        return {
            'total': len(df),
            'human': human_count,
            'robot': robot_count,
            'avg_confidence': df['average_confidence'].mean()
        }