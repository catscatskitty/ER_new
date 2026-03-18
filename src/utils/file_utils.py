"""
Утилиты для работы с файлами
"""

import os
import json
import yaml
import shutil
from pathlib import Path
import pandas as pd
from datetime import datetime
import pickle


class FileManager:
    """Класс для управления файлами"""
    
    def __init__(self, root_dir='.'):
        self.root_dir = Path(root_dir)
    
    def ensure_dir(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        return path
    
    def get_audio_files(self, directory, extensions=['.wav', '.mp3', '.ogg', '.flac', '.m4a']):
        directory = Path(directory)
        audio_files = []
        for ext in extensions:
            audio_files.extend(directory.rglob(f'*{ext}'))
            audio_files.extend(directory.rglob(f'*{ext.upper()}'))
        return sorted(list(set(audio_files)))
    
    def save_json(self, data, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_json(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_yaml(self, data, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    def load_yaml(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def save_pickle(self, data, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_pickle(self, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def save_dataframe(self, df, filepath, index=False):
        if filepath.endswith('.csv'):
            df.to_csv(filepath, index=index, encoding='utf-8')
        elif filepath.endswith('.xlsx'):
            df.to_excel(filepath, index=index)
        else:
            df.to_csv(filepath, index=index, encoding='utf-8')
    
    def copy_file(self, src, dst):
        shutil.copy2(src, dst)
    
    def move_file(self, src, dst):
        shutil.move(src, dst)
    
    def get_file_size_mb(self, filepath):
        return os.path.getsize(filepath) / (1024 * 1024)