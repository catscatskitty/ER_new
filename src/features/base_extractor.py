from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, List

class BaseExtractor(ABC):
    """
    Базовый класс для всех экстракторов признаков
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_names = []
        
    @abstractmethod
    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Извлечение признаков из аудио
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Получение названий признаков
        """
        pass
    
    def normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Нормализация признаков
        """
        return (features - np.mean(features)) / (np.std(features) + 1e-8)