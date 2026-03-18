"""
Загрузчик конфигурационных файлов
"""

import os
from pathlib import Path
from typing import Dict, Any
import yaml


class ConfigLoader:
    """Класс для загрузки конфигураций"""
    
    def __init__(self, config_dir='configs'):
        self.config_dir = Path(config_dir)
        
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Загрузка конфигурационного файла
        """
        if not config_name.endswith('.yaml') and not config_name.endswith('.yml'):
            config_name += '.yaml'
        
        config_path = self.config_dir / config_name
        
        if not config_path.exists():
            print(f"⚠️ Конфиг не найден: {config_path}, возвращаю пустой словарь")
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config if config is not None else {}
        except Exception as e:
            print(f"❌ Ошибка загрузки конфига {config_path}: {e}")
            return {}
    
    def load_all_configs(self) -> Dict[str, Any]:
        """Загрузка всех основных конфигураций"""
        configs = {}
        main_configs = ['paths_config', 'data_config', 'feature_config', 'models_config', 'training_config']
        
        for config_name in main_configs:
            try:
                configs[config_name.replace('_config', '')] = self.load_config(config_name)
            except Exception as e:
                print(f"⚠️ Ошибка загрузки {config_name}: {e}")
        
        return configs
    
    def save_config(self, config: Dict[str, Any], config_name: str):
        """Сохранение конфигурации"""
        if not config_name.endswith('.yaml') and not config_name.endswith('.yml'):
            config_name += '.yaml'
        
        config_path = self.config_dir / config_name
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)