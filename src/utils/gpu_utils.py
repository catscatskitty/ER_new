import torch
import subprocess
import sys


def check_gpu():
    """Проверка доступности GPU"""
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   PyTorch version: {torch.__version__}")
        return True
    else:
        print("❌ CUDA not available, using CPU")
        return False


def get_device(force_cpu=False):
    """Получение устройства для PyTorch"""
    if force_cpu or not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device('cuda')


def get_gpu_memory():
    """Получение информации о GPU памяти"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return {'allocated_gb': allocated, 'reserved_gb': reserved}
    return None


def print_gpu_info():
    """Вывод информации о GPU"""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")