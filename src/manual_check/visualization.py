import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import plotly.graph_objects as go
from pathlib import Path


class Visualizer:
    """Визуализация для Streamlit интерфейса"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_waveform(self, audio_path, sr=8000):
        """Построение волновой формы"""
        try:
            y, sr = librosa.load(audio_path, sr=sr)
            
            fig, ax = plt.subplots(figsize=(10, 3))
            time = np.linspace(0, len(y) / sr, len(y))
            ax.plot(time, y, color='#2E86AB', alpha=0.8, linewidth=0.8)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title('Waveform')
            ax.set_xlim(0, len(y) / sr)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error plotting waveform: {e}")
            return None
    
    def plot_spectrogram(self, audio_path, sr=8000):
        """Построение спектрограммы"""
        try:
            y, sr = librosa.load(audio_path, sr=sr)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax)
            ax.set_title('Spectrogram')
            plt.colorbar(img, ax=ax, format='%+2.0f dB')
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error plotting spectrogram: {e}")
            return None
    
    def plot_confidence_bars(self, results):
        """Построение графика уверенности моделей (Plotly)"""
        model_names = []
        confidences = []
        predictions = []
        
        for name, pred in results['model_predictions'].items():
            model_names.append(name)
            confidences.append(pred['confidence'] * 100)
            predictions.append(pred['prediction'])
        
        colors = ['#2E86AB' if p == 'human' else '#A23B72' for p in predictions]
        
        fig = go.Figure(data=[
            go.Bar(
                x=model_names,
                y=confidences,
                marker_color=colors,
                text=[f'{c:.1f}%' for c in confidences],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title='Model Confidence',
            xaxis_title='Model',
            yaxis_title='Confidence (%)',
            yaxis_range=[0, 100],
            height=400
        )
        
        return fig
    
    def plot_confusion_matrix(self, cm, labels=['Human', 'Robot']):
        """Построение матрицы ошибок"""
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=labels, yticklabels=labels,
               xlabel='Predicted', ylabel='True')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        return fig