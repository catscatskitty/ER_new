import streamlit as st
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np
import traceback

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
        except Exception:
            self.models_root = Path('results/trained_models')

        self.manual_checks_dir = Path('results/manual_checks')
        self.manual_checks_dir.mkdir(parents=True, exist_ok=True)
        Path('temp').mkdir(exist_ok=True)

        self.visualizer = Visualizer()
        self.history_manager = HistoryManager(self.manual_checks_dir)

        self.init_session_state()

    def init_session_state(self):
        defaults = {
            'current_audio': None,
            'current_results': None,
            'audio_name': None,
            'model_type': 'neural',
            'use_phonetic': False,
            'neural_processor': None,
            'traditional_processor': None,
            'trimodal_processor': None,
            'models_loaded': False,
            'loaded_models_list': []
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

    def reset_models(self):
        """Сбросить все процессоры и состояние загрузки"""
        st.session_state.neural_processor = None
        st.session_state.traditional_processor = None
        st.session_state.trimodal_processor = None
        st.session_state.models_loaded = False
        st.session_state.loaded_models_list = []

    def load_models(self):
        """Загружает модели для выбранного типа"""
        model_type = st.session_state.model_type
        use_phonetic = st.session_state.use_phonetic

        with st.spinner(f"Загрузка моделей ({model_type})..."):
            try:
                if model_type == 'neural':
                    processor = AudioProcessor(self.models_root, use_phonetic)
                    if processor and processor.models:
                        st.session_state.neural_processor = processor
                        st.session_state.loaded_models_list = list(processor.models.keys())
                        st.session_state.models_loaded = True
                        st.success(f"Загружено нейросетевых моделей: {len(processor.models)}")
                    else:
                        st.error("Нейросетевые модели не найдены. Проверьте папку results/trained_models/torch_models/")
                        self.reset_models()

                elif model_type == 'traditional':
                    processor = TraditionalModelProcessor(self.models_root)
                    if processor and processor.models:
                        st.session_state.traditional_processor = processor
                        st.session_state.loaded_models_list = list(processor.models.keys())
                        st.session_state.models_loaded = True
                        st.success(f"Загружено традиционных моделей: {len(processor.models)}")
                    else:
                        st.error("Традиционные модели не найдены. Проверьте папки results/trained_models/logistic/, random_forest/, xgboost/, catboost/")
                        self.reset_models()

                else:  # trimodal
                    processor = TriModalProcessor(self.models_root)
                    if processor and processor.model:
                        st.session_state.trimodal_processor = processor
                        st.session_state.loaded_models_list = ['trimodal']
                        st.session_state.models_loaded = True
                        st.success("Трёхмодальная модель загружена")
                    else:
                        st.error("Трёхмодальная модель не найдена. Проверьте results/trained_models/trimodal/best_trimodal.pth")
                        self.reset_models()

            except Exception as e:
                st.error(f"Ошибка загрузки: {e}")
                traceback.print_exc()
                self.reset_models()

    def get_processor(self):
        """Возвращает активный процессор, если он загружен и соответствует текущему типу"""
        if not st.session_state.models_loaded:
            return None
        model_type = st.session_state.model_type
        if model_type == 'neural':
            return st.session_state.neural_processor
        elif model_type == 'traditional':
            return st.session_state.traditional_processor
        else:
            return st.session_state.trimodal_processor

    def run(self):
        # ---- Боковая панель ----
        with st.sidebar:
            st.header("Настройки")

            model_options = {
                'neural': '🧠 Нейросетевые (акустика)',
                'traditional': '📊 Традиционные ML',
                'trimodal': '🎵 Трёхмодальная (спектрограммы+MFCC+фонетика)'
            }
            # Используем отдельный ключ, чтобы избежать конфликтов
            selected_type = st.selectbox(
                "Тип модели",
                options=list(model_options.keys()),
                format_func=lambda x: model_options[x],
                index=list(model_options.keys()).index(st.session_state.model_type),
                key="model_type_select"
            )

            # При смене типа полностью сбрасываем всё состояние загрузки
            if selected_type != st.session_state.model_type:
                st.session_state.model_type = selected_type
                self.reset_models()
                st.rerun()

            # Опция фонетики только для нейросетевых
            if st.session_state.model_type == 'neural':
                use_phonetic = st.checkbox(
                    "Использовать фонетические признаки (медленнее)",
                    value=st.session_state.use_phonetic,
                    key="use_phonetic_checkbox"
                )
                if use_phonetic != st.session_state.use_phonetic:
                    st.session_state.use_phonetic = use_phonetic
                    # При смене фонетики модели нужно перезагрузить
                    if st.session_state.models_loaded:
                        self.reset_models()
                    st.rerun()

            # Кнопка загрузки моделей
            if st.button("Загрузить модели", use_container_width=True):
                self.load_models()
                st.rerun()

            # Отображение загруженных моделей
            if st.session_state.models_loaded and st.session_state.loaded_models_list:
                st.markdown("---")
                st.subheader("Загруженные модели")
                for model_name in st.session_state.loaded_models_list:
                    display_name = {
                        'cnn': 'CNN', 'lstm': 'LSTM', 'hybrid': 'Hybrid', 'mlp': 'MLP',
                        'logistic': 'Logistic Regression', 'random_forest': 'Random Forest',
                        'xgboost': 'XGBoost', 'catboost': 'CatBoost',
                        'trimodal': 'Трёхмодальная'
                    }.get(model_name, model_name)
                    st.markdown(f"- {display_name}")
                if st.session_state.model_type == 'neural' and st.session_state.use_phonetic:
                    st.info("📝 Используются фонетические признаки")
            else:
                st.info("Модели не загружены. Нажмите кнопку выше.")

            # Справка по путям
            with st.expander("Где искать модели"):
                st.markdown(f"""
                **Нейросетевые:** `{self.models_root}/torch_models/`  
                **Традиционные:** `{self.models_root}/logistic/`, `random_forest/`, `xgboost/`, `catboost/`  
                **Трёхмодальная:** `{self.models_root}/trimodal/best_trimodal.pth`
                """)

            # Статистика
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

        # ---- Основная область ----
        if not st.session_state.models_loaded:
            st.info("Для начала работы загрузите модели в боковой панели.")
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
        uploaded_file = st.file_uploader(
            "Выберите файл",
            type=['wav', 'mp3', 'ogg', 'flac', 'm4a'],
            key="uploader"
        )
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
            accept_multiple_files=True,
            key="batch_uploader"
        )
        if uploaded_files and st.button("Запустить обработку", type="primary"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            processor = self.get_processor()
            if processor is None:
                st.error("Процессор не загружен. Попробуйте перезагрузить модели.")
                return

            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Обработка {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                temp_path = Path('temp') / uploaded_file.name
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                result = processor.classify_audio(temp_path)
                if result:
                    results.append({
                        'Файл': uploaded_file.name,
                        'Предсказание': 'Человек' if result['final_prediction'] == 'human' else 'Робот',
                        'Уверенность': f"{result['average_confidence']:.1%}"
                    })
                else:
                    results.append({
                        'Файл': uploaded_file.name,
                        'Предсказание': 'Ошибка',
                        'Уверенность': '-'
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
        """Обработка текущего аудио (загруженного или записанного)"""
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
                    processor = self.get_processor()
                    if processor is None:
                        st.error("Процессор не загружен. Проверьте, что модели загружены.")
                        return
                    try:
                        result = processor.classify_audio(st.session_state.current_audio)
                        if result:
                            st.session_state.current_results = result
                            self.display_results(result)
                            self.history_manager.add_entry(st.session_state.audio_name, result)
                        else:
                            st.error("Не удалось классифицировать аудио")
                    except Exception as e:
                        st.error(f"Ошибка при классификации: {e}")
                        traceback.print_exc()

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
        for model_key, pred in result['model_predictions'].items():
            pretty_name = {
                'cnn': 'CNN', 'lstm': 'LSTM', 'hybrid': 'Hybrid', 'mlp': 'MLP',
                'logistic': 'Logistic Regression', 'random_forest': 'Random Forest',
                'xgboost': 'XGBoost', 'catboost': 'CatBoost',
                'trimodal': 'Трёхмодальная'
            }.get(model_key, model_key)

            details.append({
                'Модель': pretty_name,
                'Предсказание': '👤 Человек' if pred['prediction'] == 'human' else '🤖 Робот',
                'Уверенность': f"{pred['confidence']:.1%}"
            })
        st.dataframe(pd.DataFrame(details), use_container_width=True)

        fig = self.visualizer.plot_confidence_bars(result)
        if fig:
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    app = ManualCheckerApp()
    app.run()