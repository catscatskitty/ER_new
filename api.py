import os
import sys
import json

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import librosa
import numpy as np
import tempfile

import threading
import time
from src.utils import ScalerManager
from src.features import extract_features
from src.models_ml import MLModelSuite
try:
    import sounddevice as sd
    HAS_SD = True
except (ImportError, OSError):
    HAS_SD = False
    print("Warning: sounddevice (PortAudio) not found. Falling back to 'arecord' on Linux.")

import subprocess

app = FastAPI(title="VoiceShield Desktop API", version="1.1.0")

SETTINGS_FILE = resource_path("settings.json")
DEFAULT_SETTINGS = {
    "threshold": 0.70,
    "audio_device": -1,
    "notifications": True
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            return {**DEFAULT_SETTINGS, **json.load(f)}
    return DEFAULT_SETTINGS

def save_settings(new_settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(new_settings, f)

current_settings = load_settings()

class LiveMonitor:
    def __init__(self):
        self.active = False
        self.latest_prob = 0.0
        self.latest_wave = []
        self.thread = None
        self.stop_event = threading.Event()

    def worker(self):
        sr = 8000
        chunk_duration = 2
        samples_needed = sr * chunk_duration
        
        import platform
        is_macos = platform.system() == 'Darwin'
        
        while not self.stop_event.is_set():
            try:
                audio = None
                device_id = current_settings.get("audio_device", -1)
                
                # macOS: используем sounddevice для всех устройств включая BlackHole/Soundflower
                if is_macos and HAS_SD:
                    try:
                        device = None
                        if isinstance(device_id, int) and device_id >= 0:
                            device = device_id
                        elif isinstance(device_id, str):
                            devices = sd.query_devices()
                            for i, d in enumerate(devices):
                                if device_id.lower() in d['name'].lower():
                                    device = i
                                    break
                        rec = sd.rec(int(samples_needed), samplerate=sr, channels=1, device=device)
                        sd.wait()
                        audio = rec.flatten()
                    except Exception as e:
                        print(f"[macOS sounddevice] ошибка: {e}")
                        time.sleep(1)
                        continue
                
                # Linux: используем parec для PulseAudio monitor или arecord для ALSA
                elif not is_macos and os.name != 'nt':
                    if HAS_SD and (not isinstance(device_id, str) or not device_id.endswith(".monitor")):
                        rec = sd.rec(int(samples_needed), samplerate=sr, channels=1)
                        sd.wait()
                        audio = rec.flatten()
                    else:
                        raw_data = None
                        
                        if isinstance(device_id, str) and device_id.endswith(".monitor"):
                            try:
                                safe_device = device_id.replace("'", "'\\''")
                                cmd = f"parec --device='{safe_device}' --rate=8000 --channels=1 --format=s16le 2>/dev/null"
                                process = subprocess.Popen(
                                    ["timeout", str(chunk_duration + 1), "bash", "-c", cmd],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE
                                )
                                raw_data, err = process.communicate(timeout=chunk_duration + 2)
                                
                                if process.returncode == 124:
                                    print(f"[parec] timeout - возможно, нет звука в устройстве")
                                elif process.returncode != 0:
                                    print(f"[parec] ошибка (код {process.returncode}): {err.decode() if err else 'неизвестно'}")
                            except subprocess.TimeoutExpired:
                                print(f"[parec] принудительно завершен по таймауту")
                                process.kill()
                                process.wait()
                            except Exception as e:
                                print(f"[parec] исключение: {e}")
                        else:
                            fallback_devices = [device_id, "default", "pulse", "plughw:0,0", "plughw:1,0"]
                            
                            for dev in fallback_devices:
                                if dev == -1: continue
                                try:
                                    cmd = ["arecord", "-d", str(chunk_duration), "-r", "8000", "-f", "S16_LE", "-t", "raw", "-q"]
                                    if dev != "default":
                                        cmd += ["-D", str(dev) if isinstance(dev, str) else f"plughw:{dev},0"]
                                    
                                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                                    raw_data, err = process.communicate(timeout=chunk_duration + 1)
                                    if raw_data and len(raw_data) > 1000:
                                        break
                                except:
                                    continue
                        
                        if raw_data and len(raw_data) > 1000:
                            audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Windows: используем sounddevice
                elif os.name == 'nt' and HAS_SD:
                    rec = sd.rec(int(samples_needed), samplerate=sr, channels=1)
                    sd.wait()
                    audio = rec.flatten()
                else:
                    time.sleep(2)
                    continue
                
                if audio is not None and len(audio) > 0:
                    rms = np.sqrt(np.mean(audio**2))
                    meter = "*" * int(rms * 100)
                    print(f"\r[УРОВЕНЬ МИКРОФОНA]: {meter:<50} ({rms:.4f})", end="")
                    
                    step = len(audio) // 100
                    if step > 0:
                        self.latest_wave = audio[::step].tolist()
                else:
                    if not (is_macos or os.name == 'nt'):
                        print("\n[!] Не удалось захватить аудио. Проверьте устройство и разрешения.")
                        time.sleep(2)

                if audio is not None and len(audio) > 0:
                    feats = extract_features(audio)
                    if not np.isnan(feats).any():
                        mean = np.array(scaler['mean'])
                        std = np.array(scaler['std'])
                        scaled = (feats - mean) / (std + 1e-8)
                        pred, conf = ml_suite.predict("xgboost", scaled.reshape(1, -1), feature_type="mfcc176")
                        
                        threshold = current_settings.get("threshold", 0.70)
                        self.latest_prob = float(conf[1]) if pred is not None else 0.0
            except Exception as e:
                print(f"Исключение в фоновом мониторинге: {e}")
                time.sleep(1)

    def start(self):
        if self.active: return
        self.active = True
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.active = False
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)

monitor = LiveMonitor()

try:
    ml_suite = MLModelSuite(model_dir=resource_path("data/models/ml"))
    scaler = None
    for scaler_path in ["data/models/ml/scaler.pkl", "data/models/scaler.pkl"]:
        scaler = ScalerManager(scaler_path).load()
        if scaler is not None:
            print(f"Scaler loaded from: {scaler_path}")
            break
    if scaler is None:
        print("Warning: No scaler found! Please train the model first.")
except Exception as e:
    print(f"Ошибка загрузки моделей или scaler: {e}")

def classify_audio_file(file_path: str) -> dict:
    start_time = time.time()
    try:
        feats = extract_features(file_path)
        if np.isnan(feats).any():
            return {"error": "Не удалось извлечь признаки из аудио"}
        scaled_feats = (feats - scaler['mean']) / (np.array(scaler['std']) + 1e-8)
        X_ml = scaled_feats.reshape(1, -1)
        pred, conf = ml_suite.predict("xgboost", X_ml, feature_type="mfcc176")
        if pred is None:
            return {"error": "Ошибка инференса модели"}
        prob = float(conf[1])
        threshold = current_settings.get("threshold", 0.50)
        is_bot = prob > threshold
        latency_ms = (time.time() - start_time) * 1000
        return {"is_bot": is_bot, "probability": prob, "latency_ms": round(latency_ms, 2)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict")
async def predict_voice(audio_file: UploadFile = File(...)):
    if not audio_file.filename.lower().endswith(('.wav', '.mp3', '.m4a')):
        raise HTTPException(status_code=400, detail="Поддерживаются только форматы: .wav, .mp3, .m4a")
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as tmp:
        contents = await audio_file.read()
        tmp.write(contents)
        tmp_path = tmp.name
    try:
        result = classify_audio_file(tmp_path)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return JSONResponse(content=result)
    except HTTPException:
        raise
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.post("/predict_batch")
async def predict_batch(audio_files: list[UploadFile] = File(...)):
    results = []
    temp_files = []
    
    if not audio_files:
        raise HTTPException(status_code=400, detail="Необходимо передать хотя бы один аудиофайл")
    
    for audio_file in audio_files:
        if not audio_file.filename.lower().endswith(('.wav', '.mp3', '.m4a')):
            results.append({
                "filename": audio_file.filename, 
                "error": "Неподдерживаемый формат файла. Разрешены: .wav, .mp3, .m4a"
            })
            continue
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as tmp:
                contents = await audio_file.read()
                tmp.write(contents)
                temp_files.append(tmp.name)
            
            result = classify_audio_file(tmp.name)
            result["filename"] = audio_file.filename
            results.append(result)
        except Exception as e:
            results.append({
                "filename": audio_file.filename,
                "error": f"Ошибка обработки файла: {str(e)}"
            })
    
    for tmp_path in temp_files:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    return JSONResponse(content={
        "results": results,
        "total_files": len(results),
        "processed": len([r for r in results if "error" not in r]),
        "errors": len([r for r in results if "error" in r])
    })

@app.post("/compare_models")
async def compare_models(audio_file: UploadFile = File(...)):
    if not audio_file.filename.lower().endswith(('.wav', '.mp3', '.m4a')):
        raise HTTPException(status_code=400, detail="Поддерживаются только форматы: .wav, .mp3, .m4a")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as tmp:
        contents = await audio_file.read()
        tmp.write(contents)
        tmp_path = tmp.name
    
    try:
        feats = extract_features(tmp_path)
        if np.isnan(feats).any():
            raise ValueError("Не удалось извлечь признаки из аудио")
        
        feature_sets = {
            "mfcc40": feats[:40],
            "mfcc80": feats[:80],
            "mfcc120": feats[:120],
            "mfcc176": feats[:176]
        }
        
        results = {}
        bot_count = 0
        human_count = 0
        
        mean = np.array(scaler['mean'])
        std = np.array(scaler['std'])
        
        for ftype, subset in feature_sets.items():
            try:
                subset_mean = mean[:len(subset)]
                subset_std = std[:len(subset)]
                scaled = (subset - subset_mean) / (subset_std + 1e-8)
                
                pred, conf = ml_suite.predict("xgboost", scaled.reshape(1, -1), feature_type=ftype)
                
                if pred is not None and conf is not None:
                    prob = float(conf[1])
                    is_bot = pred == 1
                    verdict = "БОТ" if is_bot else "ЧЕЛОВЕК"
                    if is_bot:
                        bot_count += 1
                    else:
                        human_count += 1
                    
                    results[ftype] = {
                        "verdict": verdict,
                        "probability": round(prob, 4),
                        "is_bot": is_bot
                    }
                else:
                    results[ftype] = {
                        "verdict": "ОШИБКА",
                        "probability": 0.0,
                        "is_bot": None,
                        "error": "Модель вернула None"
                    }
            except Exception as e:
                results[ftype] = {
                    "verdict": "ОШИБКА",
                    "probability": 0.0,
                    "is_bot": None,
                    "error": str(e)
                }
        
        consensus = "БОТ" if bot_count > human_count else "ЧЕЛОВЕК"
        
        valid_results = {k: v for k, v in results.items() if v.get("verdict") != "ОШИБКА"}
        if valid_results:
            recommendation = max(valid_results.items(), key=lambda x: abs(x[1]["probability"] - 0.5))[0]
        else:
            recommendation = "mfcc176"
        
        return JSONResponse(content={
            "file": audio_file.filename,
            "results": results,
            "consensus": consensus,
            "recommendation": recommendation,
            "bot_count": bot_count,
            "human_count": human_count
        })
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.get("/download_csv")
async def download_csv(results_json: str):
    import csv
    import io
    
    try:
        results_data = json.loads(results_json)
        if isinstance(results_data, dict) and "results" in results_data:
            results = results_data["results"]
        elif isinstance(results_data, list):
            results = results_data
        else:
            raise ValueError("Неверный формат данных")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Некорректный JSON результатов: {str(e)}")
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow(["Файл", "Вердикт", "Вероятность_бота (%)", "Время_обработки (мс)", "Ошибка"])
    
    for result in results:
        filename = result.get("filename", "Неизвестно")
        is_bot = result.get("is_bot", None)
        probability = round(result.get("probability", 0) * 100, 1) if "probability" in result else "N/A"
        latency = result.get("latency_ms", "N/A")
        error = result.get("error", "")
        
        if error:
            verdict = "ОШИБКА"
        elif isinstance(is_bot, bool):
            verdict = "БОТ" if is_bot else "ЧЕЛОВЕК"
        else:
            verdict = "N/A"
        
        writer.writerow([filename, verdict, probability, latency, error])
    
    csv_content = output.getvalue()
    
    temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    temp_csv.write(csv_content.encode('utf-8-sig'))
    temp_csv.close()
    
    return FileResponse(
        path=temp_csv.name,
        media_type="text/csv",
        filename="результаты_классификации_голоса.csv",
        headers={"Content-Disposition": "attachment; filename=rezultaty_klassifikatsii_golosa.csv"}
    )

@app.post("/start_live_analysis")
async def start_live():
    monitor.start()
    return {"status": "started"}

@app.post("/stop_live_analysis")
async def stop_live():
    monitor.stop()
    return {"status": "stopped"}

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = np.array([], dtype=np.float32)
    chunk_size = 8000 * 2
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_buffer = np.append(audio_buffer, chunk)
            
            if len(audio_buffer) >= chunk_size:
                analysis_chunk = audio_buffer[-chunk_size:]
                
                feats = extract_features(analysis_chunk)
                if not np.isnan(feats).any():
                    mean = np.array(scaler['mean'])
                    std = np.array(scaler['std'])
                    scaled = (feats - mean) / (std + 1e-8)
                    
                    pred, conf = ml_suite.predict("xgboost", scaled.reshape(1, -1), feature_type="mfcc176")
                    
                    prob = float(conf[1])
                    threshold = current_settings.get("threshold", 0.50)
                    await websocket.send_json({
                        "is_bot": prob > threshold,
                        "probability": prob,
                        "verdict": "БОТ" if prob > threshold else "ЧЕЛОВЕК"
                    })
                
                audio_buffer = audio_buffer[-8000:]
                
    except WebSocketDisconnect:
        print("Соединение с телефонией разорвано.")
    except Exception as e:
        print(f"Ошибка WebSocket: {e}")
    finally:
        await websocket.close()

@app.get("/get_latest_verdict")
async def get_verdict():
    try:
        threshold = current_settings.get("threshold", 0.50)
        prob = monitor.latest_prob
        is_bot = prob > threshold
        return {
            "active": monitor.active,
            "probability": round(prob, 4),
            "is_bot": is_bot,
            "verdict": "БОТ" if is_bot else "ЧЕЛОВЕК",
            "wave": monitor.latest_wave,
            "threshold": threshold
        }
    except Exception as e:
        return {"active": False, "error": str(e)}

@app.get("/get_settings")
async def get_settings():
    return load_settings()

@app.post("/save_settings")
async def update_settings(settings: dict):
    global current_settings
    current_settings = {**DEFAULT_SETTINGS, **settings}
    save_settings(current_settings)
    return {"status": "success"}

@app.get("/get_devices")
async def get_devices():
    devices = []
    
    if HAS_SD:
        try:
            for i, dev in enumerate(sd.query_devices()):
                if dev['max_input_channels'] > 0:
                    devices.append({
                        "id": i, 
                        "name": dev['name'],
                        "inputs": dev['max_input_channels']
                    })
            if devices:
                return devices
        except Exception as e:
            pass
    
    if os.name != 'nt':
        try:
            out = subprocess.check_output(["pactl", "list", "sources", "short"]).decode()
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    source_name = parts[1]
                    if source_name.endswith(".monitor"):
                        devices.append({
                            "id": source_name,
                            "name": f"Monitor: {source_name.replace('.monitor', '')}",
                            "inputs": 1
                        })
        except:
            pass
        
        try:
            out = subprocess.check_output(["pactl", "list", "sinks", "short"]).decode()
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    sink_name = parts[1]
                    monitor_name = f"{sink_name}.monitor"
                    if not any(d.get("id") == monitor_name for d in devices):
                        devices.append({
                            "id": monitor_name,
                            "name": f"Monitor: {sink_name}",
                            "inputs": 1
                        })
        except:
            pass
        
        try:
            out = subprocess.check_output(["arecord", "-L"]).decode()
            for line in out.splitlines():
                if line.startswith("plughw:") or line.startswith("default"):
                    dev_id = line.split(":")[0] if ":" in line else line
                    if not any(d.get("id") == dev_id for d in devices):
                        devices.append({
                            "id": dev_id,
                            "name": line.strip(),
                            "inputs": 1
                        })
        except:
            pass
    
    if not devices:
        devices.append({"id": -1, "name": "Устройство по умолчанию (arecord)"})
    
    return devices

STATIC_DIR = resource_path("static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def read_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(index_path)

@app.get("/{file_path:path}")
async def serve_file(file_path: str):
    if file_path.startswith(("predict", "start_live", "stop_live", "get_latest")):
        return JSONResponse(status_code=404, content={"detail": "Not found"})
    
    p = os.path.join(STATIC_DIR, file_path)
    if os.path.exists(p) and os.path.isfile(p):
        return FileResponse(p)
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

try:
    ml_suite = MLModelSuite(model_dir=resource_path("data/models/ml"))
    scaler = None
    for scaler_path in [resource_path("data/models/ml/scaler.pkl"), resource_path("data/models/scaler.pkl")]:
        scaler = ScalerManager(scaler_path).load()
        if scaler is not None:
            print(f"Scaler loaded from: {scaler_path}")
            break
    if scaler is None:
        print("Warning: No scaler found! Please train the model first.")
except Exception as e:
    print(f"Ошибка загрузки моделей или scaler: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=28465)