import numpy as np
from pathlib import Path

data_dir = Path('data/processed')
ac = np.load(data_dir / 'features_train.npy')
ph = np.load(data_dir / 'phonetic_train.npy')
print(ac.shape)  # ожидается (91813, 38)
print(ph.shape)  # ожидается (91813, 37), если для всех