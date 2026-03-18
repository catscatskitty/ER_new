"""
Модуль для логирования
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name, log_file=None, level=logging.INFO):
    """Настройка логгера"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class ExperimentLogger:
    """Класс для логирования экспериментов"""
    
    def __init__(self, experiment_name, log_dir='logs'):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f'{experiment_name}_{timestamp}.log'
        
        self.logger = setup_logger(f'{experiment_name}_{timestamp}', log_file=log_file)
        self.logger.info(f"Начат эксперимент: {experiment_name}")
        
    def log_params(self, params):
        self.logger.info("Параметры эксперимента:")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_metric(self, metric_name, value, step=None):
        if step is not None:
            self.logger.info(f"Step {step} - {metric_name}: {value:.4f}")
        else:
            self.logger.info(f"{metric_name}: {value:.4f}")
    
    def log_metrics(self, metrics, step=None):
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def log_message(self, message, level='info'):
        if level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        else:
            self.logger.debug(message)
    
    def close(self):
        self.logger.info(f"Эксперимент {self.experiment_name} завершен")
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)