import pandas as pd
import os
import shutil
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm

class AudioOrganizer:
    """
    Класс для организации аудиофайлов по папкам human/robot
    """
    
    def __init__(self, csv_path="data/raw_annotations.csv", audio_base="data/audio"):
        """
        Параметры:
        csv_path: путь к CSV файлу с аннотациями (по умолчанию data/raw_annotations.csv)
        audio_base: базовая папка с аудиофайлами (по умолчанию data/audio)
        """
        self.csv_path = Path(csv_path)
        self.audio_base = Path(audio_base)
        self.human_folder = self.audio_base / "human"
        self.robot_folder = self.audio_base / "robot"
        
        # Настройка логирования
        self.setup_logging()
        
    def setup_logging(self):
        """Настройка логирования"""
        log_filename = f"audio_organization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            encoding='utf-8'
        )
        self.logger = logging.getLogger(__name__)
        
        # Также выводим в консоль
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        console.setFormatter(formatter)
        self.logger.addHandler(console)
    
    def create_folders(self):
        """Создание необходимых папок"""
        self.human_folder.mkdir(parents=True, exist_ok=True)
        self.robot_folder.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Созданы папки:")
        self.logger.info(f"  - {self.human_folder} (для человеческой речи)")
        self.logger.info(f"  - {self.robot_folder} (для синтезированной речи)")
    
    def load_annotations(self):
        """Загрузка и проверка аннотаций"""
        try:
            self.logger.info(f"Загрузка файла: {self.csv_path}")
            
            # Проверяем существование файла
            if not self.csv_path.exists():
                self.logger.error(f"Файл не найден: {self.csv_path}")
                self.logger.info("Поиск CSV файла в папке data/...")
                
                # Ищем все CSV файлы в папке data
                data_folder = Path("data")
                csv_files = list(data_folder.glob("*.csv"))
                
                if csv_files:
                    self.csv_path = csv_files[0]
                    self.logger.info(f"Найден альтернативный файл: {self.csv_path}")
                else:
                    raise FileNotFoundError(f"CSV файл не найден в папке data/")
            
            # Загружаем CSV
            self.df = pd.read_csv(self.csv_path)
            self.logger.info(f"Загружено {len(self.df)} записей из {self.csv_path}")
            
            # Показываем первые несколько строк для проверки
            self.logger.info(f"Первые 3 записи:")
            for idx, row in self.df.head(3).iterrows():
                self.logger.info(f"  {idx}: audio_path={row.get('audio_path', 'N/A')}, authenticity={row.get('authenticity', 'N/A')}")
            
            # Проверяем наличие необходимых колонок
            required_columns = ['audio_path', 'authenticity']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            
            if missing_columns:
                self.logger.error(f"Отсутствуют колонки: {missing_columns}")
                self.logger.info(f"Доступные колонки: {list(self.df.columns)}")
                return False
            
            # Проверяем значения в колонке authenticity
            valid_values = ['real', 'fake']
            unique_values = self.df['authenticity'].unique()
            
            self.logger.info(f"Уникальные значения в authenticity: {unique_values}")
            
            # Статистика по типам
            self.stats = self.df['authenticity'].value_counts().to_dict()
            self.logger.info(f"Распределение: {self.stats}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки аннотаций: {e}")
            return False
    
    def build_file_paths(self):
        """Построение полных путей к файлам"""
        self.file_operations = []
        
        for idx, row in self.df.iterrows():
            # Получаем имя файла из пути
            audio_path = row['audio_path']
            
            if pd.isna(audio_path):
                self.logger.warning(f"Пропущена запись {idx}: audio_path is NaN")
                continue
                
            # Извлекаем имя файла (последняя часть пути)
            audio_path_str = str(audio_path)
            
            # Разные возможные форматы пути
            if '/' in audio_path_str:
                filename = audio_path_str.split('/')[-1]
            elif '\\' in audio_path_str:
                filename = audio_path_str.split('\\')[-1]
            else:
                filename = audio_path_str
            
            # Полный путь к исходному файлу
            # Ищем файл в разных местах
            possible_source_paths = [
                self.audio_base / filename,  # data/audio/filename
                Path("data") / filename,      # data/filename
                Path("audio") / filename,     # audio/filename
                Path(filename),                # filename
                self.audio_base / audio_path_str,  # data/audio/полный_путь
                Path(audio_path_str)           # исходный путь как есть
            ]
            
            # Выбираем первый существующий путь
            source = None
            for path in possible_source_paths:
                if path.exists():
                    source = path
                    break
            
            if source is None:
                source = self.audio_base / filename  # по умолчанию
            
            # Определяем целевую папку
            authenticity = str(row['authenticity']).lower().strip()
            
            if authenticity == 'real':
                destination = self.human_folder / filename
                target_type = 'human'
            else:  # 'fake' или другое
                destination = self.robot_folder / filename
                target_type = 'robot'
            
            self.file_operations.append({
                'source': source,
                'destination': destination,
                'type': target_type,
                'original_path': audio_path_str,
                'filename': filename,
                'authenticity': authenticity
            })
    
    def check_files_exist(self):
        """Проверка существования файлов"""
        self.existing_files = []
        self.missing_files = []
        
        for op in self.file_operations:
            if op['source'].exists():
                self.existing_files.append(op)
            else:
                self.missing_files.append(op)
        
        self.logger.info(f"Найдено файлов: {len(self.existing_files)}")
        self.logger.info(f"Отсутствует файлов: {len(self.missing_files)}")
        
        if self.missing_files:
            self.logger.warning("Примеры отсутствующих файлов (первые 5):")
            for op in self.missing_files[:5]:
                self.logger.warning(f"  - {op['source']} (из пути: {op['original_path']})")
    
    def organize_files(self, move=True, dry_run=False):
        """
        Организация файлов по папкам
        
        Параметры:
        move: если True - перемещает, если False - копирует
        dry_run: если True - только показывает что будет сделано
        """
        if dry_run:
            self.logger.info("\n" + "="*60)
            self.logger.info("РЕЖИМ ПРОГОНА (файлы не перемещаются)")
            self.logger.info("="*60)
        
        # Статистика операций
        human_count = 0
        robot_count = 0
        errors = []
        
        # Создаем прогресс-бар
        with tqdm(total=len(self.existing_files), desc="Обработка файлов", unit="файл") as pbar:
            for op in self.existing_files:
                try:
                    if dry_run:
                        # В режиме прогона только показываем
                        self.logger.info(f"[DRY RUN] {op['type']}: {op['filename']} -> {op['destination']}")
                        if op['type'] == 'human':
                            human_count += 1
                        else:
                            robot_count += 1
                    else:
                        # Реальное перемещение/копирование
                        if move:
                            shutil.move(str(op['source']), str(op['destination']))
                            operation = "перемещен"
                        else:
                            shutil.copy2(str(op['source']), str(op['destination']))
                            operation = "скопирован"
                        
                        # Логируем каждые 50 файлов
                        if (human_count + robot_count) % 50 == 0:
                            self.logger.info(f"{operation}: {op['filename']} -> {op['type']}")
                        
                        if op['type'] == 'human':
                            human_count += 1
                        else:
                            robot_count += 1
                            
                except Exception as e:
                    error_msg = f"Ошибка при обработке {op['filename']}: {e}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)
                
                pbar.update(1)
                pbar.set_postfix({
                    'human': human_count,
                    'robot': robot_count,
                    'errors': len(errors)
                })
        
        return human_count, robot_count, errors
    
    def generate_report(self, human_count, robot_count, errors):
        """Генерация отчета"""
        report_path = self.audio_base / "organization_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("ОТЧЕТ ОБ ОРГАНИЗАЦИИ АУДИОФАЙЛОВ\n")
            f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            f.write("ИНФОРМАЦИЯ О ФАЙЛАХ:\n")
            f.write(f"  CSV файл: {self.csv_path}\n")
            f.write(f"  Папка с аудио: {self.audio_base}\n\n")
            
            f.write("СТАТИСТИКА ИЗ АННОТАЦИЙ:\n")
            f.write(f"  Всего записей: {len(self.df)}\n")
            for key, value in self.stats.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            f.write("РЕЗУЛЬТАТЫ ОБРАБОТКИ:\n")
            f.write(f"  Файлов найдено: {len(self.existing_files)}\n")
            f.write(f"  Файлов не найдено: {len(self.missing_files)}\n")
            f.write(f"  В папку human: {human_count}\n")
            f.write(f"  В папку robot: {robot_count}\n\n")
            
            if self.missing_files:
                f.write("ПРОПУЩЕННЫЕ ФАЙЛЫ (не найдены):\n")
                for op in self.missing_files:
                    f.write(f"  - {op['filename']} (искали в: {op['source']}, путь в CSV: {op['original_path']})\n")
                f.write("\n")
            
            if errors:
                f.write("ОШИБКИ:\n")
                for error in errors:
                    f.write(f"  - {error}\n")
        
        self.logger.info(f"Отчет сохранен в: {report_path}")
        return report_path
    
    def run(self, move=True, dry_run=False):
        """
        Запуск полного процесса организации
        """
        print("\n" + "="*60)
        print("ОРГАНИЗАЦИЯ АУДИОФАЙЛОВ ПО ПАПКАМ HUMAN/ROBOT")
        print("="*60)
        
        # Создаем папки
        self.create_folders()
        
        # Загружаем аннотации
        if not self.load_annotations():
            return False
        
        # Строим пути
        self.build_file_paths()
        
        # Проверяем существование файлов
        self.check_files_exist()
        
        if not self.existing_files:
            self.logger.error("Нет файлов для обработки!")
            return False
        
        # Запрашиваем подтверждение
        if not dry_run:
            print(f"\nБудет обработано файлов: {len(self.existing_files)}")
            print(f"  human: {sum(1 for op in self.existing_files if op['type'] == 'human')}")
            print(f"  robot: {sum(1 for op in self.existing_files if op['type'] == 'robot')}")
            
            action = "ПЕРЕМЕЩЕНЫ" if move else "СКОПИРОВАНЫ"
            response = input(f"\nФайлы будут {action} в соответствующие папки. Продолжить? (д/н): ")
            
            if response.lower() != 'д':
                self.logger.info("Операция отменена пользователем")
                return False
        
        # Выполняем организацию
        human_count, robot_count, errors = self.organize_files(move, dry_run)
        
        # Генерируем отчет
        report_path = self.generate_report(human_count, robot_count, errors)
        
        # Итоговая статистика
        print("\n" + "="*60)
        print("ОПЕРАЦИЯ ЗАВЕРШЕНА")
        print("="*60)
        print(f"Обработано файлов: {human_count + robot_count}")
        print(f"  human: {human_count}")
        print(f"  robot: {robot_count}")
        print(f"  ошибок: {len(errors)}")
        print(f"  пропущено (файлы не найдены): {len(self.missing_files)}")
        print(f"\nОтчет: {report_path}")
        print("="*60)
        
        return True

def main():
    """
    Основная функция
    """
    # Пути к файлам
    CSV_PATH = "data/raw_annotations.csv"
    AUDIO_BASE = "data/audio"
    
    # Проверяем существование папок
    print("Проверка путей:")
    print(f"  CSV путь: {CSV_PATH}")
    print(f"  Аудио папка: {AUDIO_BASE}")
    print(f"  Существование CSV: {Path(CSV_PATH).exists()}")
    print(f"  Существование audio: {Path(AUDIO_BASE).exists()}")
    
    # Создаем организатор
    organizer = AudioOrganizer(
        csv_path=CSV_PATH,
        audio_base=AUDIO_BASE
    )
    
    # Запускаем
    organizer.run(move=True, dry_run=False)

if __name__ == "__main__":
    main()