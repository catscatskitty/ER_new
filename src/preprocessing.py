import numpy as np
import librosa
import soundfile as sf
import os
import glob
from sklearn.model_selection import train_test_split
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import random
import scipy.signal as signal

def apply_reverb(y, sr):
    try:
        room_size = np.random.uniform(0.1, 0.9)
        damping = np.random.uniform(0.1, 0.9)
        wet_level = np.random.uniform(0.1, 0.4)
        dry_level = 1.0 - wet_level
        
        rt60 = room_size * 2.0
        sample_rt60 = int(rt60 * sr)
        impulse = np.zeros(sample_rt60 + 1)
        impulse[0] = 1.0
        
        num_reflections = random.randint(50, 200)
        for _ in range(num_reflections):
            delay = random.randint(1, sample_rt60)
            decay = np.exp(-delay / (sample_rt60 * damping + 1)) * wet_level
            if delay < len(impulse):
                impulse[delay] += decay * np.random.uniform(-1, 1)
        
        return signal.fftconvolve(y, impulse, mode='same')
    except:
        return y

def apply_compression(y, sr):
    try:
        threshold = np.random.uniform(-30, -10)
        ratio = np.random.uniform(2, 8)
        attack = np.random.uniform(0.001, 0.01)
        release = np.random.uniform(0.01, 0.1)
        
        rms = np.sqrt(librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]**2)
        rms_db = 20 * np.log10(rms + 1e-10)
        
        gain_reduction = np.zeros_like(rms_db)
        above_threshold = rms_db > threshold
        gain_reduction[above_threshold] = (rms_db[above_threshold] - threshold) * (1 - 1/ratio)
        
        gain_db = -gain_reduction
        gain_linear = 10 ** (gain_db / 20)
        
        hop_samples = 512
        compressed = np.copy(y)
        for i, g in enumerate(gain_linear):
            start = i * hop_samples
            end = min(start + hop_samples, len(y))
            if start < len(y):
                window = np.hanning(min(hop_samples, len(y) - start))
                compressed[start:end] *= (1 - window) + window * g
        
        return np.clip(compressed, -1, 1)
    except:
        return y

def apply_background_noise(y, sr):
    try:
        noise_types = ['white', 'pink', 'brown', 'street', 'office']
        noise_type = random.choice(noise_types)
        
        if noise_type == 'white':
            noise = np.random.randn(len(y))
        elif noise_type == 'pink':
            noise = librosa.color.noise(color='pink', num_samples=len(y))
        elif noise_type == 'brown':
            noise = librosa.color.noise(color='brown', num_samples=len(y))
        else:
            noise = np.random.randn(len(y)) * 0.5
            noise = librosa.effects.pitch_shift(noise, sr=sr, n_steps=random.randint(-5, 5))
        
        noise_level = np.random.uniform(0.001, 0.05)
        noise = noise * noise_level
        
        return y + noise
    except:
        return y

def apply_eq(y, sr):
    try:
        num_bands = random.randint(3, 6)
        frequencies = np.random.uniform(100, 8000, num_bands)
        gains = np.random.uniform(-12, 12, num_bands)
        q_factors = np.random.uniform(0.5, 2.0, num_bands)
        
        for freq, gain, q in zip(frequencies, gains, q_factors):
            if freq < sr / 2:
                b, a = signal.iirpeak(freq / (sr / 2), Q=q)
                y = signal.lfilter(b, a, y)
        
        return y
    except:
        return y

def apply_bit_reduction(y, sr):
    try:
        bits = random.choice([8, 12, 16])
        max_val = 2 ** (bits - 1) - 1
        y_quantized = np.round(y * max_val) / max_val
        return y_quantized
    except:
        return y

def apply_mp3_compression(y, sr):
    try:
        bitrate = random.choice([64, 96, 128, 192, 256])
        
        temp_path = f"/tmp/mp3_test_{random.randint(0, 10000)}.mp3"
        sf.write(temp_path, y, sr)
        
        import subprocess
        cmd = ["ffmpeg", "-i", temp_path, "-b:a", f"{bitrate}k", "-y", temp_path]
        subprocess.run(cmd, capture_output=True, quiet=True)
        
        y_compressed, _ = librosa.load(temp_path, sr=sr)
        os.remove(temp_path)
        
        return y_compressed
    except:
        return y

def apply_telephony_effects(y, sr):
    try:
        y = librosa.resample(y, orig_sr=sr, target_sr=8000)
        
        nyquist = 4000
        low = 300 / nyquist
        high = 3400 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        y = signal.lfilter(b, a, y)
        
        y = librosa.effects.pitch_shift(y, sr=8000, n_steps=random.uniform(-1.5, 1.5))
        
        y = apply_compression(y, 8000)
        
        noise_level = random.uniform(0.001, 0.01)
        y = y + np.random.randn(len(y)) * noise_level
        
        if random.random() > 0.5:
            y = apply_reverb(y, 8000)
        
        return np.clip(y, -1, 1)
    except:
        return y

def apply_g711_codec(y, sr):
    try:
        y = np.clip(y, -1, 1)
        y_quantized = np.round(y * 127) / 127
        return y_quantized.astype(np.float32)
    except:
        return y

def apply_office_environment(y, sr):
    try:
        office_noise = np.zeros(len(y))
        
        t = np.arange(len(y)) / sr
        hum = 0.02 * np.sin(2 * np.pi * 60 * t)
        office_noise += hum
        
        office_noise += np.random.randn(len(y)) * 0.005
        
        num_clicks = random.randint(0, 5)
        for _ in range(num_clicks):
            click_pos = random.randint(0, len(y) - 100)
            click_len = random.randint(10, 100)
            office_noise[click_pos:click_pos+click_len] += np.random.randn(click_len) * 0.1
        
        mix_level = random.uniform(0.01, 0.05)
        return y + office_noise * mix_level
    except:
        return y

def process_single_file(file_info, processed_dir, subset):
    f, l = file_info
    try:
        out_path = os.path.join(processed_dir, subset)
        os.makedirs(out_path, exist_ok=True)
        
        if f.lower().endswith('.wav'):
            data, sr = sf.read(f)
            if sr != 8000:
                data = librosa.resample(data, orig_sr=sr, target_sr=8000)
                sr = 8000
        else:
            data, sr = librosa.load(f, sr=8000)

        base_name = os.path.basename(f)
        name, ext = os.path.splitext(base_name)
        
        sf.write(os.path.join(out_path, f"{l}_{name}_orig.wav"), data, 8000)
        
        if random.random() > 0.1:
            tele = apply_telephony_effects(data, sr)
            sf.write(os.path.join(out_path, f"{l}_{name}_tele.wav"), tele, 8000)
        
        rate = np.random.uniform(0.9, 1.1)
        speedy = librosa.effects.time_stretch(data, rate=rate)
        if (l == 1 and random.random() > 0.4) or (l == 0 and random.random() > 0.75):
            n_steps = random.choice([-3, -2, -1, 1, 2, 3])
            speedy = librosa.effects.pitch_shift(speedy, sr=sr, n_steps=n_steps)
        speedy = apply_telephony_effects(speedy, sr)
        sf.write(os.path.join(out_path, f"{l}_{name}_mixed.wav"), np.clip(speedy, -1.0, 1.0), 8000)
        
        if random.random() > 0.2:
            reverb = apply_reverb(data, sr)
            sf.write(os.path.join(out_path, f"{l}_{name}_reverb.wav"), reverb, 8000)
        
        if random.random() > 0.2:
            if random.random() > 0.5:
                noisy = apply_office_environment(data, sr)
            else:
                noisy = apply_background_noise(data, sr)
            sf.write(os.path.join(out_path, f"{l}_{name}_noise.wav"), noisy, 8000)
        
        if random.random() > 0.1:
            compressed = apply_compression(data, sr)
            sf.write(os.path.join(out_path, f"{l}_{name}_compressed.wav"), compressed, 8000)
        
        if random.random() > 0.5:
            codec = apply_g711_codec(data, sr)
            sf.write(os.path.join(out_path, f"{l}_{name}_codec.wav"), codec, 8000)
        
        if random.random() > 0.3:
            office = apply_office_environment(data, sr)
            sf.write(os.path.join(out_path, f"{l}_{name}_office.wav"), office, 8000)
        
        return True
    except Exception as e:
        print(f"Error processing {f}: {e}")
        return False

def is_already_processed(file_path, label, processed_dir):
    base_name = os.path.basename(file_path)
    target_name = f"{label}_{os.path.splitext(base_name)[0]}_orig.wav"
    for subset in ["train", "val", "test"]:
        full_path = os.path.join(processed_dir, subset, target_name)
        if os.path.exists(full_path) and os.path.getsize(full_path) > 1000:
            return True
    return False

def prepare_dataset(raw_dir, processed_dir, split=(0.7, 0.15, 0.15)):
    human_files = glob.glob(os.path.join(raw_dir, "human", "*.wav")) + glob.glob(os.path.join(raw_dir, "human", "*.mp3"))
    robot_files = glob.glob(os.path.join(raw_dir, "robot", "*.wav")) + glob.glob(os.path.join(raw_dir, "robot", "*.mp3"))
    
    new_files = []
    new_labels = []
    for f in human_files:
        if not is_already_processed(f, 0, processed_dir):
            new_files.append(f); new_labels.append(0)
    for f in robot_files:
        if not is_already_processed(f, 1, processed_dir):
            new_files.append(f); new_labels.append(1)
            
    if not new_files:
        print("No new files to process.")
        return

    print(f"Found {len(new_files)} new files. Processing with advanced augmentation...")
    try:
        train_f, test_f, train_l, test_l = train_test_split(new_files, new_labels, test_size=split[2], stratify=new_labels, random_state=42)
        train_f, val_f, train_l, val_l = train_test_split(train_f, train_l, test_size=split[1]/(split[0]+split[1]), stratify=train_l, random_state=42)
    except:
        train_f, test_f, train_l, test_l = train_test_split(new_files, new_labels, test_size=0.2, random_state=42)
        val_f, val_l = [], []

    import multiprocessing
    max_workers = 6
    for files, labels, subset in [(train_f, train_l, "train"), (val_f, val_l, "val"), (test_f, test_l, "test")]:
        if not files: continue
        print(f"Processing {subset}...")
        file_infos = list(zip(files, labels))
        with multiprocessing.Pool(processes=max_workers) as pool:
            pool.map(partial(process_single_file, processed_dir=processed_dir, subset=subset), file_infos, chunksize=20)