import streamlit as st
import librosa
import tempfile
from pathlib import Path

st.set_page_config(page_title="Загрузка файла", page_icon="📁")

st.title("📁 Загрузка аудиофайла")
st.markdown("---")

# Получаем процессоры из session state
if 'processors' not in st.session_state:
    st.error("Пожалуйста, запустите приложение через главную страницу")
    st.stop()

processors = st.session_state.processors

# Загрузка файла
uploaded_file = st.file_uploader(
    "Выберите аудиофайл для анализа",
    type=['wav', 'mp3', 'ogg', 'm4a', 'flac'],
    help="Поддерживаются форматы: WAV, MP3, OGG, M4A, FLAC"
)

if uploaded_file:
    # Сохраняем во временный файл
    with tempfile.NamedTemporaryFile(suffix=Path(uploaded_file.name).suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = tmp.name
    
    # Загружаем аудио
    try:
        audio, sr = librosa.load(temp_path, sr=8000)
        duration = len(audio) / sr
        
        st.success(f"Файл загружен: {uploaded_file.name}")
        st.info(f"Длительность: {duration:.2f} секунд")
        
        # Сохраняем в session state
        st.session_state['current_audio'] = audio
        st.session_state['current_filename'] = uploaded_file.name
        
        # Кнопка для анализа
        if st.button("🔍 Анализировать", type="primary"):
            st.switch_page("src/manual_check/app.py")
            
    except Exception as e:
        st.error(f"Ошибка при загрузке аудио: {e}")
    
    # Удаляем временный файл
    Path(temp_path).unlink()

# Информация о поддерживаемых форматах
with st.expander("ℹ️ Информация о форматах"):
    st.markdown("""
    ### Поддерживаемые форматы
    - **WAV** - без потерь, наилучшее качество
    - **MP3** - сжатие с потерями
    - **OGG** - открытый формат
    - **M4A** - AAC сжатие
    - **FLAC** - без потерь
    
    ### Рекомендации
    - Длительность: до 10 секунд
    - Частота дискретизации: 8000 Гц (конвертируется автоматически)
    - Размер файла: до 10 МБ
    """)

# Кнопка назад
if st.button("← Назад"):
    st.switch_page("src/manual_check/app.py")