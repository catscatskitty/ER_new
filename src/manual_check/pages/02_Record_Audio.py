import streamlit as st
import numpy as np
import tempfile
import soundfile as sf
from streamlit_audio_recorder import audio_recorder

st.set_page_config(page_title="Запись аудио", page_icon="🎙️")

st.title("🎙️ Запись с микрофона")
st.markdown("---")

# Проверка наличия микрофона
st.info("Нажмите кнопку ниже и начните говорить. Максимальная длительность - 10 секунд.")

# Запись аудио
audio_bytes = audio_recorder(
    text="Нажмите для записи",
    recording_color="#ff4b4b",
    neutral_color="#6c757d",
    icon_size="2x"
)

if audio_bytes:
    # Сохраняем во временный файл
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(audio_bytes)
        temp_path = tmp.name
    
    try:
        # Загружаем аудио
        import librosa
        audio, sr = librosa.load(temp_path, sr=8000)
        duration = len(audio) / sr
        
        if duration > 10:
            st.warning(f"Длительность {duration:.2f}с превышает рекомендуемую (10с). Аудио будет обрезано.")
            audio = audio[:10*8000]
            duration = 10
        
        st.success(f"Аудио записано! Длительность: {duration:.2f} секунд")
        
        # Плеер
        st.audio(audio_bytes, format="audio/wav")
        
        # Сохраняем в session state
        st.session_state['current_audio'] = audio
        st.session_state['current_filename'] = f"recording_{len(audio)}.wav"
        
        # Кнопка для анализа
        if st.button("🔍 Анализировать", type="primary"):
            st.switch_page("src/manual_check/app.py")
            
    except Exception as e:
        st.error(f"Ошибка при обработке аудио: {e}")
    
    # Удаляем временный файл
    from pathlib import Path
    Path(temp_path).unlink()

# Инструкция
with st.expander("📝 Инструкция по записи"):
    st.markdown("""
    1. Разрешите доступ к микрофону в браузере
    2. Нажмите красную кнопку для начала записи
    3. Говорите четко в микрофон
    4. Нажмите кнопку еще раз для остановки
    5. Дождитесь обработки
    
    ### Советы для лучшего качества
    - Говорите на расстоянии 10-15 см от микрофона
    - Избегайте фонового шума
    - Говорите в нормальном темпе
    - Четко произносите слова
    """)

# Кнопка назад
if st.button("← Назад"):
    st.switch_page("src/manual_check/app.py")