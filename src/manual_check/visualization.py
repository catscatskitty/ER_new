"""
Визуализация для интерфейса
"""

import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import plotly.graph_objects as go


class Visualizer:
    def plot_waveform(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            fig, ax = plt.subplots(figsize=(10, 3))
            time = np.linspace(0, len(y) / sr, len(y))
            ax.plot(time, y, color='blue', alpha=0.7, linewidth=0.5)
            ax.set_xlabel('Время (с)')
            ax.set_ylabel('Амплитуда')
            ax.set_title('Волновая форма')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Ошибка: {e}")
            return None
    
    def plot_spectrogram(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            fig, ax = plt.subplots(figsize=(10, 4))
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='viridis')
            ax.set_title('Спектрограмма')
            plt.colorbar(img, ax=ax, format='%+2.0f dB')
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Ошибка: {e}")
            return None
    
    def plot_confidence_bars(self, results):
        model_names = list(results['model_predictions'].keys())
        human_confs = []
        robot_confs = []
        
        for model_name in model_names:
            pred = results['model_predictions'][model_name]
            if pred['probabilities']:
                human_confs.append(pred['probabilities'][0])
                robot_confs.append(pred['probabilities'][1])
        
        fig = go.Figure(data=[
            go.Bar(name='Человек', x=model_names, y=human_confs,
                   marker_color='green', text=[f'{c:.1%}' for c in human_confs],
                   textposition='auto'),
            go.Bar(name='Робот', x=model_names, y=robot_confs,
                   marker_color='red', text=[f'{c:.1%}' for c in robot_confs],
                   textposition='auto')
        ])
        
        fig.update_layout(
            title='Уверенность моделей по классам',
            xaxis_title='Модель',
            yaxis_title='Вероятность',
            yaxis=dict(range=[0, 1]),
            barmode='group'
        )
        
        return fig