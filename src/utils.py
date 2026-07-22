import logging
import os
import json
import numpy as np
import warnings
import pickle

warnings.filterwarnings("ignore", category=UserWarning, message=".*hipBLASLt.*")
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

def setup_logging(log_file="pipeline.log"):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger("HumanVsRobot")

class ScalerManager:
    def __init__(self, save_path="data/models/scaler.pkl"):
        self.save_path = save_path
        self.mean = None
        self.std = None

    def load(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, 'rb') as f:
                data = pickle.load(f)
                if hasattr(data, 'mean_'):
                    self.mean = data.mean_
                    self.std = data.scale_
                else:
                    self.mean = np.array(data['mean'])
                    self.std = np.array(data['std'])
            return {
                "mean": self.mean,
                "std": self.std
            }
        return None

def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)