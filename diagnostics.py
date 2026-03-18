#!/usr/bin/env python3
"""
ПОЛНЫЙ ДИАГНОСТИЧЕСКИЙ СКРИПТ
Проверяет все компоненты системы классификации "Человек vs Робот"
Запуск: python diagnostics.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import librosa
import soundfile as sf
import json
import joblib
from tabulate import tabulate

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent))

# Цвета для вывода
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")

def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠️  {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.FAIL}❌ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")

class Diagnostics:
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.results_dir = self.root_dir / 'results'
        self.data_dir = self.root_dir / 'data'
        self.config_dir = self.root_dir / 'configs'
        self.errors = []
        self.warnings = []
        
    def run_all(self):
        """Запуск всех диагностических проверок"""
        print_header("ПОЛНАЯ ДИАГНОСТИКА СИСТЕМЫ")
        
        self.check_directory_structure()
        self.check_config_files()
        self.check_trained_models()
        self.check_feature_consistency()
        self.check_audio_files()
        self.check_model_predictions()
        self.check_gpu_availability()
        
        self.print_summary()
        
    def check_directory_structure(self):
        """Проверка структуры директорий"""
        print_header("1. ПРОВЕРКА СТРУКТУРЫ ДИРЕКТОРИЙ")
        
        required_dirs = [
            ('data/audio/human', 'Папка с человеческой речью'),
            ('data/audio/robot', 'Папка с синтезированной речью'),
            ('data/processed', 'Обработанные данные'),
            ('data/processed/augmented_8khz', 'Аугментированные данные'),
            ('data/splits', 'Разбиение данных'),
            ('results/trained_models', 'Обученные модели'),
            ('results/metrics', 'Метрики'),
            ('results/plots', 'Графики'),
            ('logs', 'Логи'),
            ('temp', 'Временные файлы'),
        ]
        
        for dir_path, description in required_dirs:
            full_path = self.root_dir / dir_path
            if full_path.exists():
                files = list(full_path.rglob('*'))
                print_success(f"{dir_path}/ - {description} ({len(files)} файлов)")
            else:
                print_error(f"{dir_path}/ - {description} (НЕ НАЙДЕНА)")
                self.errors.append(f"Отсутствует директория: {dir_path}")
    
    def check_config_files(self):
        """Проверка конфигурационных файлов"""
        print_header("2. ПРОВЕРКА КОНФИГУРАЦИЙ")
        
        config_files = [
            'paths_config.yaml',
            'data_config.yaml',
            'feature_config.yaml',
            'models_config.yaml',
            'training_config.yaml'
        ]
        
        import yaml
        
        for config_file in config_files:
            config_path = self.config_dir / config_file
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    print_success(f"{config_file} - загружен успешно")
                    
                    # Проверка ключевых параметров
                    if config_file == 'data_config.yaml':
                        sr = config.get('data', {}).get('sample_rate', None)
                        if sr == 8000:
                            print_success(f"  sample_rate: {sr} Гц (корректно для телефонии)")
                        else:
                            print_warning(f"  sample_rate: {sr} Гц (ожидалось 8000)")
                            
                    if config_file == 'feature_config.yaml':
                        features = config.get('features', {})
                        mfcc = features.get('mfcc', {})
                        print_info(f"  MFCC параметры: n_mfcc={mfcc.get('n_mfcc')}, n_fft={mfcc.get('n_fft')}")
                        
                except Exception as e:
                    print_error(f"{config_file} - ошибка загрузки: {e}")
                    self.errors.append(f"Ошибка в конфиге {config_file}")
            else:
                print_error(f"{config_file} - НЕ НАЙДЕН")
                self.errors.append(f"Отсутствует конфиг: {config_file}")
    
    def check_trained_models(self):
        """Проверка обученных моделей"""
        print_header("3. ПРОВЕРКА ОБУЧЕННЫХ МОДЕЛЕЙ")
        
        # Нейросетевые модели
        torch_models_dir = self.results_dir / 'trained_models' / 'torch_models'
        if torch_models_dir.exists():
            models = ['cnn', 'lstm', 'hybrid']
            for model_name in models:
                model_path = torch_models_dir / f'best_{model_name}.pth'
                if model_path.exists():
                    size = model_path.stat().st_size / 1024
                    try:
                        # Проверка целостности
                        state_dict = torch.load(model_path, map_location='cpu')
                        print_success(f"torch_models/best_{model_name}.pth - {size:.1f} KB, слоев: {len(state_dict)}")
                    except Exception as e:
                        print_error(f"torch_models/best_{model_name}.pth - поврежден: {e}")
                        self.errors.append(f"Повреждена модель {model_name}")
                else:
                    print_warning(f"torch_models/best_{model_name}.pth - НЕ НАЙДЕНА")
        
        # Традиционные модели
        traditional_models = ['logistic', 'random_forest', 'xgboost', 'catboost']
        for model_name in traditional_models:
            model_dir = self.results_dir / 'trained_models' / model_name
            model_file = model_dir / 'model.pkl'
            if model_file.exists():
                size = model_file.stat().st_size / 1024
                try:
                    data = joblib.load(model_file)
                    if isinstance(data, dict):
                        has_model = 'model' in data
                        has_scaler = 'scaler' in data
                        print_success(f"{model_name}/model.pkl - {size:.1f} KB, модель: {has_model}, scaler: {has_scaler}")
                    else:
                        print_success(f"{model_name}/model.pkl - {size:.1f} KB")
                except Exception as e:
                    print_error(f"{model_name}/model.pkl - поврежден: {e}")
                    self.errors.append(f"Повреждена модель {model_name}")
            else:
                print_warning(f"{model_name}/model.pkl - НЕ НАЙДЕНА")
    
    def check_feature_consistency(self):
        """Проверка согласованности признаков"""
        print_header("4. ПРОВЕРКА ПРИЗНАКОВ")
        
        # Загружаем признаки из датасета
        feature_files = ['features_train.npy', 'features_val.npy', 'features_test.npy']
        label_files = ['labels_train.npy', 'labels_val.npy', 'labels_test.npy']
        
        expected_dim = 38  # Ожидаемая размерность признаков
        
        for feat_file, label_file in zip(feature_files, label_files):
            feat_path = self.data_dir / 'processed' / feat_file
            label_path = self.data_dir / 'processed' / label_file
            
            if feat_path.exists() and label_path.exists():
                X = np.load(feat_path)
                y = np.load(label_path)
                
                print_info(f"\n{feat_file}:")
                print_info(f"  Форма: {X.shape}")
                print_info(f"  Тип: {X.dtype}")
                print_info(f"  Диапазон: [{X.min():.4f}, {X.max():.4f}]")
                print_info(f"  Среднее: {X.mean():.4f}, Std: {X.std():.4f}")
                print_info(f"  Human: {np.sum(y==0)}, Robot: {np.sum(y==1)}")
                
                if X.shape[1] != expected_dim:
                    print_error(f"  Размерность {X.shape[1]} != ожидаемой {expected_dim}")
                    self.errors.append(f"Неверная размерность в {feat_file}")
                else:
                    print_success(f"  Размерность корректна: {X.shape[1]}")
                    
                # Проверка на NaN
                if np.isnan(X).any():
                    print_error(f"  Обнаружены NaN значения!")
                    self.errors.append(f"NaN в {feat_file}")
                else:
                    print_success(f"  Нет NaN значений")
            else:
                print_warning(f"{feat_file} или {label_file} не найдены")
    
    def check_audio_files(self):
        """Проверка аудиофайлов"""
        print_header("5. ПРОВЕРКА АУДИОФАЙЛОВ")
        
        # Выбираем по одному файлу из каждой категории
        sample_files = []
        
        human_dir = self.data_dir / 'audio' / 'human'
        if human_dir.exists():
            human_files = list(human_dir.rglob('*.wav'))[:2]
            sample_files.extend([(f, 'human') for f in human_files])
        
        robot_dir = self.data_dir / 'audio' / 'robot'
        if robot_dir.exists():
            robot_files = list(robot_dir.rglob('*.wav'))[:2]
            sample_files.extend([(f, 'robot') for f in robot_files])
        
        augmented_dir = self.data_dir / 'processed' / 'augmented_8khz'
        if augmented_dir.exists():
            aug_files = list(augmented_dir.rglob('*.wav'))[:2]
            sample_files.extend([(f, 'augmented') for f in aug_files])
        
        for file_path, category in sample_files:
            try:
                # Загрузка с оригинальной частотой
                y, sr = librosa.load(file_path, sr=None)
                print_info(f"\n{category}: {file_path.name}")
                print_info(f"  Оригинальная частота: {sr} Гц")
                print_info(f"  Длительность: {len(y)/sr:.2f} сек")
                print_info(f"  Каналы: {'стерео' if len(y.shape) > 1 else 'моно'}")
                print_info(f"  Макс. амплитуда: {np.max(np.abs(y)):.4f}")
                
                # Проверка на тишину
                if np.max(np.abs(y)) < 0.01:
                    print_warning(f"  Файл почти пустой (тишина)")
                
                # Проверка загрузки с target_sr=8000
                y8k, sr8k = librosa.load(file_path, sr=8000)
                print_info(f"  После ресемплинга: {sr8k} Гц, длина: {len(y8k)}")
                
            except Exception as e:
                print_error(f"  Ошибка загрузки: {e}")
    
    def check_model_predictions(self):
        """Проверка предсказаний моделей на тестовых данных"""
        print_header("6. ПРОВЕРКА ПРЕДСКАЗАНИЙ МОДЕЛЕЙ")
        
        # Загружаем тестовые данные
        X_test_path = self.data_dir / 'processed' / 'features_test.npy'
        y_test_path = self.data_dir / 'processed' / 'labels_test.npy'
        
        if not (X_test_path.exists() and y_test_path.exists()):
            print_warning("Тестовые данные не найдены, пропускаем проверку предсказаний")
            return
        
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
        
        # Загружаем традиционные модели
        traditional_models = {}
        for model_name in ['logistic', 'random_forest', 'xgboost', 'catboost']:
            model_path = self.results_dir / 'trained_models' / model_name / 'model.pkl'
            if model_path.exists():
                try:
                    data = joblib.load(model_path)
                    if isinstance(data, dict):
                        model = data.get('model')
                        scaler = data.get('scaler')
                    else:
                        model = data
                        scaler = None
                    
                    if model is not None:
                        traditional_models[model_name] = (model, scaler)
                        print_success(f"Загружена {model_name}")
                except Exception as e:
                    print_error(f"Не удалось загрузить {model_name}: {e}")
        
        if traditional_models:
            print_info("\nПроверка предсказаний на первых 10 тестовых образцах:")
            
            # Берем первые 10 образцов
            X_sample = X_test[:10]
            y_sample = y_test[:10]
            
            results = []
            for i in range(10):
                row = {'#' : i+1, 'Истина': 'human' if y_sample[i]==0 else 'robot'}
                
                for name, (model, scaler) in traditional_models.items():
                    X = X_sample[i].reshape(1, -1)
                    if scaler is not None:
                        X = scaler.transform(X)
                    
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)[0]
                        pred = np.argmax(proba)
                        conf = proba[pred]
                    else:
                        pred = model.predict(X)[0]
                        conf = 1.0
                    
                    row[name] = 'human' if pred==0 else 'robot'
                    row[f'{name}_conf'] = f"{conf:.2f}"
                
                results.append(row)
            
            df = pd.DataFrame(results)
            print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    
    def check_gpu_availability(self):
        """Проверка доступности GPU"""
        print_header("7. ПРОВЕРКА GPU")
        
        if torch.cuda.is_available():
            print_success(f"CUDA доступна")
            print_info(f"  Версия CUDA: {torch.version.cuda}")
            print_info(f"  Количество GPU: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print_info(f"  GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB")
            
            # Проверка mixed precision
            if hasattr(torch.cuda, 'amp'):
                print_success(f"  Mixed precision поддерживается")
            else:
                print_warning(f"  Mixed precision не поддерживается")
        else:
            print_warning("CUDA не доступна, используется CPU")
    
    def print_summary(self):
        """Вывод итоговой сводки"""
        print_header("ИТОГОВАЯ СВОДКА")
        
        if not self.errors and not self.warnings:
            print_success("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!")
            print_info("Система должна работать корректно.")
        else:
            if self.errors:
                print_error(f"❌ НАЙДЕНО ОШИБОК: {len(self.errors)}")
                for i, error in enumerate(self.errors, 1):
                    print_error(f"  {i}. {error}")
            
            if self.warnings:
                print_warning(f"⚠️  ПРЕДУПРЕЖДЕНИЙ: {len(self.warnings)}")
                for i, warning in enumerate(self.warnings, 1):
                    print_warning(f"  {i}. {warning}")
            
            print_info("\nРекомендации:")
            print_info("1. Исправьте ошибки перед запуском")
            print_info("2. Проверьте соответствие параметров в конфигах")
            print_info("3. Убедитесь, что все модели обучены")
            print_info("4. Запустите повторную диагностику")

if __name__ == "__main__":
    # Установка tabulate если не установлен
    try:
        from tabulate import tabulate
    except ImportError:
        os.system(f"{sys.executable} -m pip install tabulate")
        from tabulate import tabulate
    
    diagnostics = Diagnostics()
    diagnostics.run_all()