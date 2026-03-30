import yaml
from pathlib import Path

class ConfigLoader:
    def __init__(self, config_dir='configs'):
        self.config_dir = Path(config_dir)
    
    def load_config(self, config_name):
        path = self.config_dir / f"{config_name}.yaml"
        with open(path, 'r') as f:
            return yaml.safe_load(f)