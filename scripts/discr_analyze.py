import librosa
import matplotlib.pyplot as plt

# Загрузить один и тот же файл робота
y_16, sr_16 = librosa.load('data/processed/augmented_8khz/robot/root_268_phone.wav', sr=16000)
y_8, sr_8 = librosa.load('data/processed/augmented_8khz/robot/root_268_phone.wav', sr=8000)

# Сравнить MFCC
mfcc_16 = librosa.feature.mfcc(y=y_16, sr=sr_16, n_mfcc=13)
mfcc_8 = librosa.feature.mfcc(y=y_8, sr=sr_8, n_mfcc=13)

# Визуализировать разницу
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
librosa.display.specshow(mfcc_16, ax=axes[0], sr=sr_16)
axes[0].set_title('16 kHz MFCC')
librosa.display.specshow(mfcc_8, ax=axes[1], sr=sr_8)
axes[1].set_title('8 kHz MFCC')
plt.show()