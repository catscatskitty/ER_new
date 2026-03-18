"""
Обучение Wav2Vec2 с поддержкой GPU и mixed precision
Путь: src/models/deep_learning/train_wave2vec2_gpu.py
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from transformers import TFWav2Vec2ForSequenceClassification, Wav2Vec2Config
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.utils.file_utils import FileManager
from src.utils.logger import setup_logger
from src.utils.config_loader import ConfigLoader
from src.utils.reproducibility import set_random_seeds
from utils.gpu_utils import (
    configure_gpu, configure_mixed_precision, 
    clear_gpu_memory, get_device_strategy
)


class Wav2Vec2DataGenerator(tf.keras.utils.Sequence):
    """Генератор данных для Wav2Vec2"""
    
    def __init__(self, file_paths, labels, batch_size=8, sr=16000, max_len=16000*5, shuffle=True):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.sr = sr
        self.max_len = max_len
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.file_paths))
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        batch_files = [self.file_paths[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]
        
        batch_audio = []
        for file_path in batch_files:
            import librosa
            y, _ = librosa.load(file_path, sr=self.sr)
            
            if len(y) > self.max_len:
                y = y[:self.max_len]
            else:
                y = np.pad(y, (0, max(0, self.max_len - len(y))), 'constant')
            
            batch_audio.append(y)
        
        X = np.array(batch_audio, dtype=np.float32)
        y = np.array(batch_labels, dtype=np.float32)
        
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


class Wav2Vec2_GPU_Trainer:
    """Класс для обучения Wav2Vec2 с поддержкой GPU"""
    
    def __init__(self, config_path='configs'):
        self.config_loader = ConfigLoader(config_path)
        self.training_config = self.config_loader.load_config('training_config')
        self.file_manager = FileManager()
        
        paths_config = self.config_loader.load_config('paths_config')
        self.models_dir = Path(paths_config['paths']['models_root']) / 'wav2vec2_gpu'
        self.metrics_dir = Path(paths_config['paths']['metrics_root'])
        self.plots_dir = Path(paths_config['paths']['plots_root'])
        self.splits_root = Path(paths_config['paths']['splits_root'])
        
        self.file_manager.ensure_dir(self.models_dir)
        self.file_manager.ensure_dir(self.metrics_dir)
        self.file_manager.ensure_dir(self.plots_dir)
        
        self.logger = setup_logger('wav2vec2_gpu_trainer')
        
        # Настройка GPU
        gpu_config = self.training_config['training'].get('gpu', {'enabled': True})
        self.gpu_available = configure_gpu(gpu_config)
        self.mixed_precision = configure_mixed_precision(self.training_config)
        
        clear_gpu_memory()
        set_random_seeds(self.training_config['training']['random_seed'])
        
        # Параметры обучения
        dl_config = self.training_config['training']['deep_learning']
        self.batch_size = 4  # Wav2Vec2 требует меньше из-за размера
        self.epochs = dl_config['epochs']
        self.learning_rate = 2e-5  # Маленький LR для fine-tuning
    
    def load_data(self):
        """Загрузка данных"""
        self.logger.info("Загрузка данных...")
        
        train_files, train_labels = [], []
        val_files, val_labels = [], []
        
        with open(self.splits_root / 'train_files.txt', 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    train_files.append(parts[0])
                    train_labels.append(int(parts[1]))
        
        with open(self.splits_root / 'val_files.txt', 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    val_files.append(parts[0])
                    val_labels.append(int(parts[1]))
        
        self.logger.info(f"Train: {len(train_files)} файлов")
        self.logger.info(f"Val: {len(val_files)} файлов")
        
        return train_files, val_files, train_labels, val_labels
    
    def create_model(self):
        """Создание модели Wav2Vec2"""
        self.logger.info("Загрузка предобученной модели Wav2Vec2...")
        
        strategy = get_device_strategy()
        
        with strategy.scope():
            # Загружаем предобученную модель
            model_name = "facebook/wav2vec2-base"
            
            model = TFWav2Vec2ForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2,
                ignore_mismatched_sizes=True
            )
            
            # Замораживаем большую часть слоев для fine-tuning
            if hasattr(model, 'wav2vec2'):
                for layer in model.wav2vec2.encoder.layers[:-2]:
                    layer.trainable = False
            
            # Компилируем
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            
            model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )
        
        self.logger.info(f"Модель загружена. Параметров: {model.count_params():,}")
        
        return model
    
    def train(self, model, train_gen, val_gen):
        """Обучение модели"""
        self.logger.info("Начало обучения...")
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.models_dir / 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                str(self.metrics_dir / 'wav2vec2_gpu_training_log.csv')
            )
        ]
        
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def run(self):
        """Запуск обучения"""
        self.logger.info("=" * 60)
        self.logger.info(" ОБУЧЕНИЕ WAV2VEC2 С ПОДДЕРЖКОЙ GPU")
        self.logger.info("=" * 60)
        
        self.logger.info(f"GPU доступен: {self.gpu_available}")
        self.logger.info(f"Mixed precision: {self.mixed_precision}")
        
        train_files, val_files, train_labels, val_labels = self.load_data()
        
        train_gen = Wav2Vec2DataGenerator(
            train_files, train_labels,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_gen = Wav2Vec2DataGenerator(
            val_files, val_labels,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        model = self.create_model()
        
        history = self.train(model, train_gen, val_gen)
        
        clear_gpu_memory()
        
        self.logger.info("✅ Wav2Vec2 (GPU) обучена")
        
        return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Обучение Wav2Vec2 с GPU')
    parser.add_argument('--config', type=str, default='configs')
    args = parser.parse_args()
    
    trainer = Wav2Vec2_GPU_Trainer(args.config)
    trainer.run()