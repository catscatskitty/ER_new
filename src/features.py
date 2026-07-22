import numpy as np
import librosa
import os
import soundfile as sf

def extract_features(audio_path_or_y, sr=8000, feature_type="mfcc176"):
    try:
        if isinstance(audio_path_or_y, str):
            y, _ = librosa.load(audio_path_or_y, sr=sr)
        else:
            y = audio_path_or_y
        
        y = librosa.util.normalize(y)
        
        if np.max(np.abs(y)) < 1e-6:
            if feature_type == "mfcc40":
                return np.zeros(40, dtype=np.float32)
            elif feature_type == "mfcc80":
                return np.zeros(80, dtype=np.float32)
            elif feature_type == "mfcc120":
                return np.zeros(120, dtype=np.float32)
            else:
                return np.zeros(176, dtype=np.float32)
        
        spec = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
        spec_db = librosa.amplitude_to_db(spec)
        
        if feature_type == "mfcc40":
            return _extract_mfcc40(y, sr, spec_db)
        elif feature_type == "mfcc80":
            return _extract_mfcc80(y, sr, spec, spec_db)
        elif feature_type == "mfcc120":
            return _extract_mfcc120(y, sr, spec, spec_db)
        else:
            return _extract_mfcc176(y, sr, spec, spec_db)
            
    except Exception as e:
        if feature_type == "mfcc40":
            return np.zeros(40, dtype=np.float32)
        elif feature_type == "mfcc80":
            return np.zeros(80, dtype=np.float32)
        elif feature_type == "mfcc120":
            return np.zeros(120, dtype=np.float32)
        else:
            return np.zeros(176, dtype=np.float32)

def _extract_mfcc40(y, sr, spec_db):
    mfccs = librosa.feature.mfcc(S=spec_db, n_mfcc=20, sr=sr)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    
    features = np.concatenate([mfcc_mean, mfcc_std]).astype(np.float32)
    if len(features) < 40:
        features = np.pad(features, (0, 40 - len(features)))
    else:
        features = features[:40]
    return features

def _extract_mfcc80(y, sr, spec, spec_db):
    mfccs = librosa.feature.mfcc(S=spec_db, n_mfcc=20, sr=sr)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    
    centroid = librosa.feature.spectral_centroid(S=spec)[0]
    rolloff = librosa.feature.spectral_rolloff(S=spec)[0]
    flatness = librosa.feature.spectral_flatness(S=spec)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    
    spectral_features = np.array([
        float(np.mean(centroid)), float(np.std(centroid)),
        float(np.mean(rolloff)), float(np.std(rolloff)),
        float(np.mean(flatness)), float(np.std(flatness)),
        float(np.mean(zcr)), float(np.std(zcr))
    ], dtype=np.float32)
    
    delta_mfcc = librosa.feature.delta(mfccs)
    delta_std = np.std(delta_mfcc, axis=1)
    
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_midi('C2'), fmax=librosa.note_to_midi('C7'),
            sr=sr, frame_length=1024, hop_length=512
        )
        f0_voiced = f0[voiced_flag]
        if len(f0_voiced) > 5:
            pitch_features = np.array([
                float(np.mean(f0_voiced)), float(np.std(f0_voiced)),
                float(np.max(f0_voiced) - np.min(f0_voiced)), float(np.median(f0_voiced)),
                float(np.sum(voiced_flag) / len(voiced_flag)),
                float(np.mean(np.abs(np.diff(f0_voiced))))
            ], dtype=np.float32)
        else:
            pitch_features = np.zeros(6, dtype=np.float32)
    except:
        pitch_features = np.zeros(6, dtype=np.float32)
    
    try:
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]
        if len(rms) > 5:
            shimmer_features = np.array([
                float(np.mean(np.abs(np.diff(rms)))),
                float(np.mean(np.abs(np.diff(20 * np.log10(rms + 1e-10))))),
                float(np.std(rms)), float(np.max(rms) - np.min(rms))
            ], dtype=np.float32)
        else:
            shimmer_features = np.zeros(4, dtype=np.float32)
    except:
        shimmer_features = np.zeros(4, dtype=np.float32)
    
    features = np.concatenate([
        mfcc_mean, mfcc_std, spectral_features, delta_std, pitch_features, shimmer_features
    ]).astype(np.float32)
    
    if len(features) < 80:
        features = np.pad(features, (0, 80 - len(features)))
    else:
        features = features[:80]
    return features

def _extract_mfcc120(y, sr, spec, spec_db):
    mfccs = librosa.feature.mfcc(S=spec_db, n_mfcc=20, sr=sr)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    
    try:
        cqt = np.abs(librosa.cqt(y=y, sr=sr, hop_length=512, n_bins=72, bins_per_octave=12, fmin=65.41))
        log_cqt = np.log10(cqt + 1e-10)
        cqcc = librosa.feature.mfcc(S=log_cqt, n_mfcc=20)
        cqcc_mean = np.mean(cqcc, axis=1)
        cqcc_std = np.std(cqcc, axis=1)
    except:
        cqcc_mean = np.zeros(20, dtype=np.float32)
        cqcc_std = np.zeros(20, dtype=np.float32)
    
    try:
        lfcc = librosa.feature.mfcc(S=spec_db, n_mfcc=20, dct_type=2)
        lfcc_mean = np.mean(lfcc, axis=1)
        lfcc_std = np.std(lfcc, axis=1)
    except:
        lfcc_mean = np.zeros(20, dtype=np.float32)
        lfcc_std = np.zeros(20, dtype=np.float32)
    
    features = np.concatenate([
        mfcc_mean, mfcc_std, cqcc_mean, cqcc_std, lfcc_mean, lfcc_std
    ]).astype(np.float32)
    
    if len(features) < 120:
        features = np.pad(features, (0, 120 - len(features)))
    else:
        features = features[:120]
    return features

def _extract_mfcc176(y, sr, spec, spec_db):
    mfccs = librosa.feature.mfcc(S=spec_db, n_mfcc=20, sr=sr)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    
    centroid = librosa.feature.spectral_centroid(S=spec)[0]
    rolloff = librosa.feature.spectral_rolloff(S=spec)[0]
    flatness = librosa.feature.spectral_flatness(S=spec)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    
    spectral_features = np.array([
        float(np.mean(centroid)), float(np.std(centroid)),
        float(np.mean(rolloff)), float(np.std(rolloff)),
        float(np.mean(flatness)), float(np.std(flatness)),
        float(np.mean(zcr)), float(np.std(zcr))
    ], dtype=np.float32)
    
    delta_mfcc = librosa.feature.delta(mfccs)
    delta_std = np.std(delta_mfcc, axis=1)
    
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_midi('C2'), fmax=librosa.note_to_midi('C7'),
            sr=sr, frame_length=1024, hop_length=512
        )
        f0_voiced = f0[voiced_flag]
        if len(f0_voiced) > 5:
            pitch_features = np.array([
                float(np.mean(f0_voiced)), float(np.std(f0_voiced)),
                float(np.max(f0_voiced) - np.min(f0_voiced)), float(np.median(f0_voiced)),
                float(np.sum(voiced_flag) / len(voiced_flag)),
                float(np.mean(np.abs(np.diff(f0_voiced))))
            ], dtype=np.float32)
        else:
            pitch_features = np.zeros(6, dtype=np.float32)
    except:
        pitch_features = np.zeros(6, dtype=np.float32)
    
    try:
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]
        if len(rms) > 5:
            shimmer_features = np.array([
                float(np.mean(np.abs(np.diff(rms)))),
                float(np.mean(np.abs(np.diff(20 * np.log10(rms + 1e-10))))),
                float(np.std(rms)), float(np.max(rms) - np.min(rms))
            ], dtype=np.float32)
        else:
            shimmer_features = np.zeros(4, dtype=np.float32)
    except:
        shimmer_features = np.zeros(4, dtype=np.float32)
    
    try:
        cqt = np.abs(librosa.cqt(y=y, sr=sr, hop_length=512, n_bins=72, bins_per_octave=12, fmin=65.41))
        log_cqt = np.log10(cqt + 1e-10)
        cqcc = librosa.feature.mfcc(S=log_cqt, n_mfcc=20)
        cqcc_mean = np.mean(cqcc, axis=1)
        cqcc_std = np.std(cqcc, axis=1)
    except:
        cqcc_mean = np.zeros(20, dtype=np.float32)
        cqcc_std = np.zeros(20, dtype=np.float32)
    
    try:
        lfcc = librosa.feature.mfcc(S=spec_db, n_mfcc=20, dct_type=2)
        lfcc_mean = np.mean(lfcc, axis=1)
        lfcc_std = np.std(lfcc, axis=1)
    except:
        lfcc_mean = np.zeros(20, dtype=np.float32)
        lfcc_std = np.zeros(20, dtype=np.float32)
    
    try:
        contrast = librosa.feature.spectral_contrast(S=spec_db, sr=sr, n_bands=7)
        contrast_mean = np.mean(contrast, axis=1)
        contrast_std = np.std(contrast, axis=1)
        
        bandwidth = librosa.feature.spectral_bandwidth(S=spec)[0]
        bandwidth_mean = float(np.mean(bandwidth))
        bandwidth_std = float(np.std(bandwidth))
        
        from scipy import stats
        spec_flat = spec_db.flatten()
        if len(spec_flat) > 10:
            skewness_val = float(stats.skew(spec_flat))
            kurtosis_val = float(stats.kurtosis(spec_flat))
        else:
            skewness_val = 0.0
            kurtosis_val = 0.0
        
        additional_features = np.array([
            contrast_mean[0], contrast_mean[1], contrast_mean[2], 
            contrast_mean[3], contrast_mean[4], contrast_mean[5], contrast_mean[6],
            contrast_std[0], contrast_std[1], contrast_std[2],
            contrast_std[3], contrast_std[4], contrast_std[5], contrast_std[6],
            bandwidth_mean, bandwidth_std,
            skewness_val, kurtosis_val
        ], dtype=np.float32)
    except:
        additional_features = np.zeros(18, dtype=np.float32)
    
    features = np.concatenate([
        mfcc_mean, mfcc_std, spectral_features, delta_std, pitch_features,
        shimmer_features, cqcc_mean, cqcc_std, lfcc_mean, lfcc_std, additional_features
    ]).astype(np.float32)
    
    if len(features) < 176:
        features = np.pad(features, (0, 176 - len(features)))
    else:
        features = features[:176]
    return features

def extract_features_fast(audio_path_or_y, sr=8000):
    try:
        if isinstance(audio_path_or_y, str):
            y, _ = librosa.load(audio_path_or_y, sr=sr)
        else:
            y = audio_path_or_y
        
        y = librosa.util.normalize(y)
        
        if np.max(np.abs(y)) < 1e-6:
            return np.zeros(40, dtype=np.float32)
        
        spec = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
        spec_db = librosa.amplitude_to_db(spec)
        
        mfccs = librosa.feature.mfcc(S=spec_db, n_mfcc=13, sr=sr)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        centroid = librosa.feature.spectral_centroid(S=spec)[0]
        rolloff = librosa.feature.spectral_rolloff(S=spec)[0]
        flatness = librosa.feature.spectral_flatness(S=spec)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        spectral_features = np.array([
            float(np.mean(centroid)), float(np.std(centroid)),
            float(np.mean(rolloff)), float(np.std(rolloff)),
            float(np.mean(flatness)), float(np.std(flatness)),
            float(np.mean(zcr)), float(np.std(zcr))
        ], dtype=np.float32)
        
        features = np.concatenate([mfcc_mean, mfcc_std, spectral_features]).astype(np.float32)
        
        if len(features) < 40:
            features = np.pad(features, (0, 40 - len(features)))
        else:
            features = features[:40]
        
        return features
        
    except Exception as e:
        return np.zeros(40, dtype=np.float32)

def extract_mfcc_features(audio_path_or_y, sr=8000):
    return extract_features(audio_path_or_y, sr=sr, feature_type="mfcc176")

def extract_all_features(audio_path_or_y, sr=8000):
    return extract_features(audio_path_or_y, sr=sr, feature_type="mfcc176")
