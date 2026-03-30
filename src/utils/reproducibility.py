import random
import numpy as np
import torch
import os


def set_seed(seed=42):
    """Фиксация random seed для воспроизводимости"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ Random seed set to {seed}")


def get_reproducible_loader(dataset, batch_size, shuffle=True, seed=42):
    """Создание DataLoader с фиксированным seed"""
    from torch.utils.data import DataLoader
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
    )