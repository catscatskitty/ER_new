import streamlit as st
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.manual_check.audio_processor import AudioProcessor
from src.manual_check.traditional_processor import TraditionalModelProcessor
from src.manual_check.trimodal_processor import TriModalProcessor
from src.manual_check.visualization import Visualizer
from src.manual_check.history_manager import HistoryManager
from src.utils.file_utils import FileManager
from src.utils.config_loader import ConfigLoader

st.set_page_config(page_title="Audio Classifier - Human vs Robot", page_icon="🎤", layout="wide")
st.title("🎤 Классификатор 'Человек vs Робот'")
st.markdown("---")


class ManualCheckerApp:
    def __init__(self):
        self.config_loader = ConfigLoader('configs')
        self.file_manager = FileManager()

        try:
            paths_config = self.config_loader.load_config('paths_config')
            self.models_root = Path(paths_config['paths']['models_root'])
        except Exception as e:
            print(f"Ошибка загрузки конфига: {e}")
            self.models_root = Path('results/trained_models')

        self.manual_checks_dir = Path('results/manual_checks')
        self.manual_checks_dir.mkdir(parents=True, exist_ok=True)
        Path('temp').mkdir(exist_ok=True)

        self.visualizer = Visualizer()
        self.history_manager = HistoryManager(self.manual_checks_dir)

        self.init_session_state()

    def init_session_state(self):
        if 'current_audio' not in st.session_state:
            st.session_state.current_audio = None
        if 'current_results' not in st.session_state:
            st.session_state.current_results = None
        if 'audio_name' not in st.session_state:
            st.session_state.audio_name = None
        if 'model_type' not in st.session_state:
            st.session_state.model_type = 'neural'
        if 'use_phonetic' not in st.session_state:
            st.session_state.use_phonetic = False
        if 'neural_processor' not in st.session_state:
            st.session_state.neural_processor = None
        if 'traditional_processor' not in st.session_state:
            st.session_state.traditional_processor = None
        if 'trimodal_processor' not in st.session_state:
            st.session_state.trimodal_processor = None
        if 'models_loaded' not in st.session_state:
            st.session_state.models_loaded = False

    def load_models(self):
        """Загружает модели для выбранного типа"""
        model_type = st.session_state.model_type
        if model_type == 'neural':
            with st.spinner("Загрузка нейросетевых моделей..."):
                processor = AudioProcessor(self.models_root, st.session_state.use_phonetic)
                if processor and processor.models:
                    st.session_state.neural_processor = processor
                    st.session_state.models_loaded = True
                    st.success(f"Загружено нейросетевых моделей: {len(processor.models)}")
                else:
                    st.error("Нейросетевые модели не найдены")
        elif model_type == 'traditional':
            with st.spinner("Загрузка традиционных моделей..."):
                processor = TraditionalModelProcessor(self.models_root)
                if processor and processor.models:
                    st.session_state.traditional_processor = processor
                    st.session_state.models_loaded = True
                    st.success(f"Загружено традиционных моделей: {len(processor.models)}")
                else:
                    st.error("Традиционные модели не найдены")
        else:  # trimodal
            with st.spinner("Загрузка трёхмодальной модели..."):
                processor = TriModalProcessor(self.models_root)
                if processor and processor.model:
                    st.session_state.trimodal_processor = processor
                    st.session_state.models_loaded = True
                    st.success("Трёхмодальная модель загружена")
                else:
                    st.error("Трёхмодальная модель не найдена")

    def get_processor(self):
        """Возвращает процессор для выбранного типа"""
        if st.session_state.model_type == 'neural':
            return st.session_state.neural_processor
        elif st.session_state.model_type == 'traditional':
            return st.session_state.traditional_processor
        else:
            return st.session_state.trimodal_processor

    def run(self):
        with st.sidebar:
            st.header("Настройки")

            options = ['neural', 'traditional', 'trimodal']
            format_func = {
                'neural': '🧠 Нейросетевые (акустика)',
                'traditional': '📊 Традиционные ML',
                'trimodal': '🎵 Трёхмодальная (спектрограммы+MFCC+фонетика)'
            }
            current_index = options.index(st.session_state.model_type)

            model_type = st.selectbox(
                "Тип модели",
                options=options,
                format_func=lambda x: format_func.get(x, x),
                index=current_index,
                key="model_type_select"
            )

            if model_type != st.session_state.model_type:
                st.session_state.model_type = model_type
                st.session_state.models_loaded = False
                # Сбрасываем процессоры, чтобы не было путаницы
                st.session_state.neural_processor = None
                st.session_state.traditional_processor = None
                st.session_state.trimodal_processor = None
                st.rerun()

            if model_type == 'neural':
                use_phonetic = st.checkbox(
                    "Использовать фонетические признаки (медленнее)",
                    value=st.session_state.use_phonetic,
                    key="use_phonetic_checkbox"
                )
                if use_phonetic != st.session_state.use_phonetic:
                    st.session_state.use_phonetic = use_phonetic
                    st.session_state.models_loaded = False
                    st.rerun()
            else:
                st.session_state.use_phonetic = False

            if st.button("Загрузить модели", use_container_width=True):
                self.load_models()
                st.rerun()

            # Отображаем информацию о моделях только для выбранного типа
            if st.session_state.models_loaded:
                if model_type == 'neural' and st.session_state.neural_processor and st.session_state.neural_processor.models:
                    st.markdown("**🧠 Нейросетевые:**")
                    for name in st.session_state.neural_processor.models.keys():
                        st.markdown(f"- {name.upper()}")
                elif model_type == 'traditional' and st.session_state.traditional_processor and st.session_state.traditional_processor.models:
                    st.markdown("**📊 Традиционные:**")
                    display = {
                        'logistic': 'Logistic Regression',
                        'random_forest': 'Random Forest',
                        'xgboost': 'XGBoost',
                        'catboost': 'CatBoost'
                    }
                    for name in st.session_state.traditional_processor.models.keys():
                        st.markdown(f"- {display.get(name, name)}")
                elif model_type == 'trimodal' and st.session_state.trimodal_processor and st.session_state.trimodal_processor.model:
                    st.markdown("**🎵 Трёхмодальная:** загружена")
                else:
                    st.info(f"Модели типа {model_type} не загружены")
            else:
                st.info("Модели не загружены")

            with st.expander("Где искать модели"):
                st.markdown(f"""
                **Нейросетевые модели:** `{self.models_root}/torch_models/`  
                **Традиционные модели:** `{self.models_root}/logistic/` `{self.models_root}/random_forest/` `{self.models_root}/xgboost/` `{self.models_root}/catboost/`  
                **Трёхмодальная модель:** `{self.models_root}/trimodal/best_trimodal.pth`
                """)

            st.markdown("---")
            st.subheader("Статистика")
            history = self.history_manager.get_history()
            if history:
                st.metric("Всего проверок", len(history))
                human = sum(1 for h in history if h.get('final_prediction') == 'human')
                robot = sum(1 for h in history if h.get('final_prediction') == 'robot')
                if human + robot > 0:
                    st.progress(human / (human + robot))
                    st.caption(f"👤 Человек: {human} | 🤖 Робот: {robot}")
            else:
                st.metric("Всего проверок", 0)

        if not st.session_state.models_loaded:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("""
                ## 👋 Добро пожаловать!
                Для начала работы загрузите модели в боковой панели.
                
                ### Поддерживаемые форматы:
                WAV, MP3, OGG, FLAC, M4A
                """)
            return

        tab1, tab2, tab3 = st.tabs(["📂 Загрузка файла", "🎙️ Запись с микрофона", "📊 Пакетная обработка"])
        with tab1:
            self.render_upload_tab()
        with tab2:
            self.render_record_tab()
        with tab3:
            self.render_batch_tab()

    def render_upload_tab(self):
        st.header("Загрузите аудиофайл для анализа")
        uploaded_file = st.file_uploader("Выберите файл", type=['wav', 'mp3', 'ogg', 'flac', 'm4a'])
        if uploaded_file is not None:
            temp_path = Path('temp') / uploaded_file.name
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.current_audio = temp_path
            st.session_state.audio_name = uploaded_file.name
            self.process_current_audio()

    def render_record_tab(self):
        st.header("Запишите аудио с микрофона")
        audio_bytes = st.audio_input("Нажмите для записи", key="audio_recorder")
        if audio_bytes:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_path = Path('temp') / f"recording_{timestamp}.wav"
            with open(temp_path, 'wb') as f:
                f.write(audio_bytes.read())
            st.session_state.current_audio = temp_path
            st.session_state.audio_name = f"recording_{timestamp}.wav"
            self.process_current_audio()

    def render_batch_tab(self):
        st.header("Пакетная обработка файлов")
        uploaded_files = st.file_uploader(
            "Выберите несколько файлов",
            type=['wav', 'mp3', 'ogg', 'flac', 'm4a'],
            accept_multiple_files=True
        )
        if uploaded_files and st.button("Запустить обработку", type="primary"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            processor = self.get_processor()
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Обработка {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                temp_path = Path('temp') / uploaded_file.name
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                result = processor.classify_audio(temp_path) if processor else None

                if result:
                    results.append({
                        'Файл': uploaded_file.name,
                        'Предсказание': 'Человек' if result['final_prediction'] == 'human' else 'Робот',
                        'Уверенность': f"{result['average_confidence']:.1%}"
                    })
                progress_bar.progress((i + 1) / len(uploaded_files))

            status_text.text("Обработка завершена!")
            if results:
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                csv_path = self.manual_checks_dir / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(csv_path, index=False, encoding='utf-8')
                st.success(f"Результаты сохранены: `{csv_path}`")

    def process_current_audio(self):
        if st.session_state.current_audio:
            st.audio(str(st.session_state.current_audio))
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Волновая форма")
                fig_wave = self.visualizer.plot_waveform(st.session_state.current_audio)
                if fig_wave:
                    st.pyplot(fig_wave)
            with col2:
                st.subheader("Спектрограмма")
                fig_spec = self.visualizer.plot_spectrogram(st.session_state.current_audio)
                if fig_spec:
                    st.pyplot(fig_spec)

            if st.button("Классифицировать", type="primary", use_container_width=True):
                with st.spinner("Анализ аудио..."):
                    try:
                        processor = self.get_processor()
                        if processor is None:
                            st.error("Процессор не загружен")
                            return

                        result = processor.classify_audio(st.session_state.current_audio)
                        if result:
                            st.session_state.current_results = result
                            self.display_results(result)
                            self.history_manager.add_entry(st.session_state.audio_name, result)
                        else:
                            st.error("Не удалось классифицировать аудио")
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        st.error(f"Ошибка: {e}")

    def display_results(self, result):
        st.markdown("---")
        st.header("Результаты классификации")

        col1, col2, col3 = st.columns(3)
        with col1:
            if result['final_prediction'] == 'human':
                st.success("### 👤 ЧЕЛОВЕК")
            else:
                st.error("### 🤖 РОБОТ")
        with col2:
            st.metric("Уверенность", f"{result['average_confidence']:.1%}")
        with col3:
            st.metric("Голосование", f"{result['human_votes']}/{result['robot_votes']}")

        st.subheader("Детальные результаты по моделям")
        details = []
        for model_name, pred in result['model_predictions'].items():
            clean = model_name.replace('neural_', '').replace('traditional_', '')
            display = {
                'cnn': 'CNN', 'lstm': 'LSTM', 'hybrid': 'Hybrid', 'mlp': 'MLP',
                'logistic': 'Logistic Regression', 'random_forest': 'Random Forest',
                'xgboost': 'XGBoost', 'catboost': 'CatBoost', 'trimodal': 'Трёхмодальная'
            }.get(clean, clean)
            details.append({
                'Модель': display,
                'Предсказание': '👤 Человек' if pred['prediction'] == 'human' else '🤖 Робот',
                'Уверенность': f"{pred['confidence']:.1%}"
            })
        st.dataframe(pd.DataFrame(details), use_container_width=True)

        st.subheader("Уверенность моделей")
        fig = self.visualizer.plot_confidence_bars(result)
        if fig:
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    app = ManualCheckerApp()
    app.run()