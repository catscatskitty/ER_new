"""
Главное приложение Streamlit
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.manual_check.audio_processor import AudioProcessor
from src.manual_check.traditional_processor import TraditionalModelProcessor
from src.manual_check.visualization import Visualizer
from src.manual_check.history_manager import HistoryManager
from src.utils.file_utils import FileManager
from src.utils.config_loader import ConfigLoader

st.set_page_config(page_title="Audio Classifier - Human vs Robot", page_icon="🎤", layout="wide")
st.title("🎤 Классификатор 'Человек vs Робот'")
st.markdown("---")

# Инициализация сессии
if 'neural_processor' not in st.session_state:
    st.session_state.neural_processor = None
if 'traditional_processor' not in st.session_state:
    st.session_state.traditional_processor = None
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = Visualizer()
if 'history_manager' not in st.session_state:
    config_loader = ConfigLoader('configs')
    try:
        paths_config = config_loader.load_config('paths_config')
        manual_checks_dir = Path(paths_config['paths']['manual_checks'])
    except:
        manual_checks_dir = Path('results/manual_checks')
    st.session_state.history_manager = HistoryManager(manual_checks_dir)
if 'models_root' not in st.session_state:
    config_loader = ConfigLoader('configs')
    try:
        paths_config = config_loader.load_config('paths_config')
        st.session_state.models_root = Path(paths_config['paths']['models_root'])
    except:
        st.session_state.models_root = Path('results/trained_models')
if 'current_audio' not in st.session_state:
    st.session_state.current_audio = None
if 'current_results' not in st.session_state:
    st.session_state.current_results = None
if 'audio_name' not in st.session_state:
    st.session_state.audio_name = None

# Создание директорий
Path('temp').mkdir(exist_ok=True)


def load_models():
    """Загрузка моделей"""
    with st.spinner("Загрузка моделей..."):
        st.session_state.neural_processor = AudioProcessor(st.session_state.models_root)
        st.session_state.traditional_processor = TraditionalModelProcessor(st.session_state.models_root)
        
        neural_count = st.session_state.neural_processor.load_models() or 0
        traditional_count = st.session_state.traditional_processor.load_models() or 0
        
        if neural_count + traditional_count > 0:
            st.success(f"Загружено моделей: {neural_count + traditional_count} (нейросетевых: {neural_count}, традиционных: {traditional_count})")
            return True
        else:
            st.error("Модели не найдены")
            return False


def classify_audio(audio_path):
    """Классификация аудио"""
    neural_result = None
    traditional_result = None
    
    if st.session_state.neural_processor and st.session_state.neural_processor.models:
        neural_result = st.session_state.neural_processor.classify_audio(audio_path)
    
    if st.session_state.traditional_processor and st.session_state.traditional_processor.models:
        traditional_result = st.session_state.traditional_processor.classify_audio(audio_path)
    
    if neural_result or traditional_result:
        return combine_results(neural_result, traditional_result)
    return None


def combine_results(neural_result=None, traditional_result=None):
    """Объединение результатов"""
    combined = {
        'model_predictions': {},
        'human_votes': 0,
        'robot_votes': 0,
        'total_confidence': 0
    }
    
    if neural_result and 'model_predictions' in neural_result:
        for model_name, pred in neural_result['model_predictions'].items():
            combined['model_predictions'][f"neural_{model_name}"] = pred
            if pred['prediction'] == 'human':
                combined['human_votes'] += 1
            else:
                combined['robot_votes'] += 1
            combined['total_confidence'] += pred['confidence']
    
    if traditional_result and 'model_predictions' in traditional_result:
        for model_name, pred in traditional_result['model_predictions'].items():
            combined['model_predictions'][f"traditional_{model_name}"] = pred
            if pred['prediction'] == 'human':
                combined['human_votes'] += 1
            else:
                combined['robot_votes'] += 1
            combined['total_confidence'] += pred['confidence']
    
    if combined['human_votes'] > combined['robot_votes']:
        combined['final_prediction'] = 'human'
    elif combined['robot_votes'] > combined['human_votes']:
        combined['final_prediction'] = 'robot'
    else:
        avg_human = get_average_confidence(combined, 'human')
        avg_robot = get_average_confidence(combined, 'robot')
        combined['final_prediction'] = 'human' if avg_human > avg_robot else 'robot'
    
    total = len(combined['model_predictions'])
    combined['average_confidence'] = combined['total_confidence'] / total if total > 0 else 0
    return combined


def get_average_confidence(results, target_class):
    """Средняя уверенность для класса"""
    confidences = [pred['confidence'] for pred in results['model_predictions'].values() if pred['prediction'] == target_class]
    return np.mean(confidences) if confidences else 0


def display_results(results):
    """Отображение результатов"""
    st.markdown("---")
    st.header("Результаты")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if results['final_prediction'] == 'human':
            st.success("### ЧЕЛОВЕК")
        else:
            st.error("### РОБОТ")
    with col2:
        st.metric("Уверенность", f"{results['average_confidence']:.1%}")
    with col3:
        st.metric("Голоса", f"{results['human_votes']}/{results['robot_votes']}")
    
    # Детали
    details = []
    for model_name, pred in results['model_predictions'].items():
        clean_name = model_name.replace('neural_', '').replace('traditional_', '')
        display_name = {
            'cnn': 'CNN', 'lstm': 'LSTM', 'hybrid': 'Hybrid',
            'logistic': 'Logistic', 'random_forest': 'Random Forest',
            'xgboost': 'XGBoost', 'catboost': 'CatBoost'
        }.get(clean_name, clean_name)
        details.append({
            'Модель': display_name,
            'Предсказание': 'Человек' if pred['prediction'] == 'human' else 'Робот',
            'Уверенность': f"{pred['confidence']:.1%}"
        })
    
    if details:
        st.dataframe(pd.DataFrame(details), use_container_width=True)
    
    # График
    fig = st.session_state.visualizer.plot_confidence_bars(results)
    if fig:
        st.plotly_chart(fig, use_container_width=True)


# Боковая панель
with st.sidebar:
    st.header("Настройки")
    
    if st.button("Загрузить модели", use_container_width=True):
        load_models()
    
    with st.expander("Где искать модели"):
        st.markdown(f"""
        **Нейросетевые:** `{st.session_state.models_root}/cnn_gpu/` `{st.session_state.models_root}/lstm_gpu/` `{st.session_state.models_root}/hybrid_gpu/`
        **Традиционные:** `{st.session_state.models_root}/logistic/` `{st.session_state.models_root}/random_forest/` `{st.session_state.models_root}/xgboost/` `{st.session_state.models_root}/catboost/`
        """)
    
    st.markdown("---")
    st.subheader("Статистика")
    history = st.session_state.history_manager.get_history()
    if history:
        st.metric("Всего проверок", len(history))
        human_count = sum(1 for h in history if h.get('final_prediction') == 'human')
        robot_count = sum(1 for h in history if h.get('final_prediction') == 'robot')
        if human_count + robot_count > 0:
            st.progress(human_count / (human_count + robot_count))
            st.caption(f"Человек: {human_count} | Робот: {robot_count}")
    else:
        st.metric("Всего проверок", 0)

# Основной контент
has_models = (st.session_state.neural_processor and len(st.session_state.neural_processor.models) > 0) or \
             (st.session_state.traditional_processor and len(st.session_state.traditional_processor.models) > 0)

if not has_models:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👆 Загрузите модели в боковой панели")
else:
    tab1, tab2, tab3 = st.tabs(["📂 Файл", "🎙️ Микрофон", "📊 Пакетная"])
    
    # Вкладка загрузки файла
    with tab1:
        st.header("Загрузите аудиофайл")
        uploaded_file = st.file_uploader("Выберите файл", type=['wav', 'mp3', 'ogg', 'flac', 'm4a'], key="file_uploader")
        
        if uploaded_file is not None:
            # Сохраняем файл
            temp_path = Path('temp') / uploaded_file.name
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            st.audio(str(temp_path))
            
            if st.button("Классифицировать", key="classify_file", use_container_width=True):
                with st.spinner("Анализ..."):
                    results = classify_audio(temp_path)
                    if results:
                        st.session_state.current_results = results
                        st.session_state.history_manager.add_entry(uploaded_file.name, results)
                        st.rerun()
            
            # Показываем результаты если есть
            if st.session_state.current_results:
                display_results(st.session_state.current_results)
    
    # Вкладка записи с микрофона
    with tab2:
        st.header("Запишите аудио")
        audio_bytes = st.audio_input("Нажмите для записи", key="audio_recorder")
        
        if audio_bytes:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_path = Path('temp') / f"recording_{timestamp}.wav"
            with open(temp_path, 'wb') as f:
                f.write(audio_bytes.read())
            
            st.audio(str(temp_path))
            
            if st.button("Классифицировать", key="classify_record", use_container_width=True):
                with st.spinner("Анализ..."):
                    results = classify_audio(temp_path)
                    if results:
                        st.session_state.current_results = results
                        st.session_state.history_manager.add_entry(f"recording_{timestamp}.wav", results)
                        st.rerun()
            
            if st.session_state.current_results:
                display_results(st.session_state.current_results)
    
    # Вкладка пакетной обработки
    with tab3:
        st.header("Пакетная обработка")
        uploaded_files = st.file_uploader("Выберите файлы", type=['wav', 'mp3', 'ogg', 'flac', 'm4a'], 
                                          accept_multiple_files=True, key="batch_uploader")
        
        if uploaded_files and st.button("Запустить обработку", type="primary", key="batch_process"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Обработка {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                
                temp_path = Path('temp') / uploaded_file.name
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                neural_result = st.session_state.neural_processor.classify_audio(temp_path) if st.session_state.neural_processor and st.session_state.neural_processor.models else None
                traditional_result = st.session_state.traditional_processor.classify_audio(temp_path) if st.session_state.traditional_processor and st.session_state.traditional_processor.models else None
                
                if neural_result or traditional_result:
                    combined = combine_results(neural_result, traditional_result)
                    results.append({
                        'Файл': uploaded_file.name,
                        'Предсказание': 'Человек' if combined['final_prediction'] == 'human' else 'Робот',
                        'Уверенность': f"{combined['average_confidence']:.1%}"
                    })
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("Готово!")
            
            if results:
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                csv_path = st.session_state.history_manager.history_dir / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(csv_path, index=False, encoding='utf-8')
                st.success(f"Результаты сохранены")