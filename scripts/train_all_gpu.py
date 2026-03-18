#!/usr/bin/env python3
"""
Скрипт для последовательного обучения всех моделей на GPU
Путь: scripts/train_all_gpu.py
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.gpu_utils import clear_gpu_memory
from src.utils.logger import setup_logger


class AllGPUTrainer:
    """Класс для последовательного обучения всех GPU моделей"""
    
    def __init__(self):
        self.logger = setup_logger('all_gpu_trainer')
        self.models = [
            {
                'name': 'CNN',
                'script': 'src/models/deep_learning/train_cnn_gpu.py',
                'enabled': True
            },
            {
                'name': 'LSTM',
                'script': 'src/models/deep_learning/train_lstm_gpu.py',
                'enabled': True
            },
            {
                'name': 'Hybrid CNN-LSTM',
                'script': 'src/models/deep_learning/train_hybrid_gpu.py',
                'enabled': True
            },
            {
                'name': 'Wav2Vec2',
                'script': 'src/models/deep_learning/train_wave2vec2_gpu.py',
                'enabled': False  # Опционально, требует много памяти
            }
        ]
    
    def run(self):
        """Запуск обучения всех моделей"""
        self.logger.info("=" * 60)
        self.logger.info("🚀 ОБУЧЕНИЕ ВСЕХ МОДЕЛЕЙ НА GPU")
        self.logger.info("=" * 60)
        
        total_start = time.time()
        
        for model in self.models:
            if not model['enabled']:
                self.logger.info(f"\n⏭️  {model['name']} - пропущен")
                continue
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🎯 ОБУЧЕНИЕ: {model['name']}")
            self.logger.info(f"{'='*60}")
            
            # Очищаем память перед каждой моделью
            clear_gpu_memory()
            
            # Запускаем обучение
            start_time = time.time()
            cmd = [sys.executable, model['script'], '--config', 'configs']
            
            import subprocess
            try:
                subprocess.run(cmd, check=True)
                elapsed = time.time() - start_time
                self.logger.info(f"✅ {model['name']} обучена за {elapsed:.1f} сек")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"❌ Ошибка при обучении {model['name']}: {e}")
            
            # Пауза между моделями
            if model != self.models[-1]:
                self.logger.info("\n⏳ Пауза 5 секунд перед следующей моделью...")
                time.sleep(5)
        
        total_time = time.time() - total_start
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"✅ ВСЕ МОДЕЛИ ОБУЧЕНЫ")
        self.logger.info(f"⏱️  Общее время: {total_time/60:.1f} минут")
        self.logger.info("=" * 60)


if __name__ == "__main__":
    trainer = AllGPUTrainer()
    trainer.run()