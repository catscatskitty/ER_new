"""
Утилиты для работы с GPU
"""

import torch
import numpy as np
import random
import gc


def setup_device(use_gpu=True, gpu_id=0):
    """Настройка устройства для PyTorch"""
    if use_gpu and torch.cuda.is_available():
        if gpu_id < torch.cuda.device_count():
            device = torch.device(f'cuda:{gpu_id}')
            print(f"Используется GPU: {torch.cuda.get_device_name(gpu_id)}")
            print(f"VRAM: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB")
        else:
            device = torch.device('cuda:0')
            print(f"GPU {gpu_id} не найден, используется GPU 0")
    else:
        device = torch.device('cpu')
        print("Используется CPU")
    
    return device


def set_random_seeds(seed=42):
    """Установка seed для воспроизводимости"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Установлен seed: {seed}")


def clear_gpu_memory():
    """Очистка памяти GPU"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("Память очищена")


def get_device():
    """Получение устройства для torch"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def get_optimal_batch_size(model_name='cnn', base_batch_size=32, mixed_precision=False):
    """Получение оптимального batch size"""
    if not torch.cuda.is_available():
        return base_batch_size // 2
    
    try:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        mp_multiplier = 2.0 if mixed_precision else 1.0
        
        model_multipliers = {
            'cnn': 1.0,
            'lstm': 0.8,
            'hybrid': 0.6,
            'transformer': 0.3
        }
        
        multiplier = model_multipliers.get(model_name, 1.0) * mp_multiplier
        
        if gpu_memory >= 24:
            return int(base_batch_size * 4 * multiplier)
        elif gpu_memory >= 16:
            return int(base_batch_size * 3 * multiplier)
        elif gpu_memory >= 8:
            return int(base_batch_size * 2 * multiplier)
        else:
            return int(base_batch_size * multiplier)
            
    except Exception as e:
        print(f"Ошибка при определении batch size: {e}")
        return base_batch_size


def print_gpu_info():
    """Вывод информации о GPU"""
    print("\nИНФОРМАЦИЯ О GPU")
    print("-" * 40)
    
    if not torch.cuda.is_available():
        print("CUDA не доступна")
        return
    
    print(f"CUDA версия: {torch.version.cuda}")
    print(f"Количество GPU: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Всего памяти: {props.total_memory / 1024**3:.1f} GB")
        
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  Использовано: {allocated:.2f} GB")
        print(f"  Зарезервировано: {reserved:.2f} GB")
    
    print("-" * 40)


class AverageMeter:
    """Хранение средних значений"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Ранняя остановка обучения"""
    def __init__(self, patience=10, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
        
        return self.early_stop