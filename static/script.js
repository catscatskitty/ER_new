document.addEventListener('DOMContentLoaded', () => {
    const navLinks = document.querySelectorAll('.nav-links li');
    const pages = document.querySelectorAll('.page');
    const startMicBtn = document.getElementById('start-mic');
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const liveVerdict = document.getElementById('live-verdict');
    const waveform = document.getElementById('waveform');
    const ctx = waveform.getContext('2d');
    const analyzeBtn = document.getElementById('analyze-btn');
    const downloadCsvBtn = document.getElementById('download-csv-btn');
    const scannerControls = document.getElementById('scanner-controls');
    const fileCount = document.getElementById('file-count');
    const compareDropZone = document.getElementById('compare-drop-zone');
    const compareFileInput = document.getElementById('compare-file-input');
    const compareResults = document.getElementById('compare-results');

    let scanResults = [];
    let selectedFiles = [];
    let compareFiles = [];
    let compareChart = null;

    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            const pageId = link.getAttribute('data-page');
            navLinks.forEach(l => l.classList.remove('active'));
            pages.forEach(p => p.classList.remove('active'));
            link.classList.add('active');
            document.getElementById(pageId).classList.add('active');
        });
    });

    let dataArray = new Array(100).fill(0);
    let isMonitoring = false;

    function draw() {
        requestAnimationFrame(draw);
        ctx.fillStyle = 'rgba(10, 11, 16, 0.2)';
        ctx.fillRect(0, 0, waveform.width, waveform.height);
        ctx.lineWidth = 3;
        ctx.strokeStyle = '#00d2ff';
        ctx.beginPath();
        const sliceWidth = waveform.width * 1.0 / dataArray.length;
        let x = 0;
        for (let i = 0; i < dataArray.length; i++) {
            const val = dataArray[i];
            const y = (val * 50) + (waveform.height / 2);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
            x += sliceWidth;
        }
        ctx.stroke();
    }

    const thresholdRange = document.getElementById('threshold-range');
    const thresholdDisplay = document.getElementById('threshold-display');
    const deviceSelect = document.getElementById('device-select');
    const notifyToggle = document.getElementById('notify-toggle');
    const saveBtn = document.getElementById('save-settings');

    thresholdRange.addEventListener('input', (e) => {
        thresholdDisplay.innerText = parseFloat(e.target.value).toFixed(2);
    });

    async function loadSettings() {
        try {
            const res = await fetch('/get_settings');
            const settings = await res.json();
            thresholdRange.value = settings.threshold;
            thresholdDisplay.innerText = settings.threshold.toFixed(2);
            notifyToggle.checked = settings.notifications;

            const devRes = await fetch('/get_devices');
            const devices = await devRes.json();
            deviceSelect.innerHTML = devices.map(d =>
                `<option value="${d.id}" ${d.id === settings.audio_device ? 'selected' : ''}>${d.name}</option>`
            ).join('');
        } catch (err) {
            console.error('Не удалось загрузить настройки:', err);
        }
    }

    saveBtn.addEventListener('click', async () => {
        const deviceValue = deviceSelect.value;
        // Определяем тип устройства: если строка с .monitor - сохраняем как строку, иначе как число
        let audioDevice;
        if (typeof deviceValue === 'string' && deviceValue.includes('.monitor')) {
            audioDevice = deviceValue;  // PulseAudio monitor - сохраняем как строку
        } else {
            audioDevice = parseInt(deviceValue) || -1;  // ALSA устройство - сохраняем как число
        }
        
        const settings = {
            threshold: parseFloat(thresholdRange.value),
            audio_device: audioDevice,
            notifications: notifyToggle.checked
        };
        try {
            await fetch('/save_settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settings)
            });
            alert('Настройки успешно сохранены!');
        } catch (err) {
            alert('Не удалось сохранить настройки.');
        }
    });

    function resizeCanvas() {
        waveform.width = waveform.offsetWidth;
        waveform.height = waveform.offsetHeight;
    }
    window.addEventListener('resize', resizeCanvas);

    loadSettings();
    resizeCanvas();
    draw();

    startMicBtn.addEventListener('click', async () => {
        isMonitoring = !isMonitoring;
        if (isMonitoring) {
            startMicBtn.innerHTML = '<i class="fas fa-stop"></i> Остановить мониторинг';
            startMicBtn.style.background = 'linear-gradient(135deg, #ff416c, #ff4b2b)';
            liveVerdict.classList.remove('hidden');
            startBackendMonitoring();
        } else {
            startMicBtn.innerHTML = '<i class="fas fa-play"></i> Запустить мониторинг';
            startMicBtn.style.background = 'linear-gradient(135deg, #00d2ff, #3a7bd5)';
            liveVerdict.classList.add('hidden');
            stopBackendMonitoring();
        }
    });

    let backendInterval;

    function startBackendMonitoring() {
        fetch('/start_live_analysis', { method: 'POST' });
        backendInterval = setInterval(async () => {
            const res = await fetch('/get_latest_verdict');
            const data = await res.json();
            if (data.active) updateVerdictUI(data);
        }, 2000);
    }

    function stopBackendMonitoring() {
        fetch('/stop_live_analysis', { method: 'POST' });
        clearInterval(backendInterval);
    }

    function updateVerdictUI(data) {
        if (data.wave && data.wave.length > 0) {
            dataArray = data.wave;
        }
        const bar = document.getElementById('bot-probability');
        const text = document.getElementById('prob-text');
        const verdict = document.getElementById('verdict-text');
        const prob = data.probability * 100;
        bar.style.width = prob + '%';
        bar.style.backgroundColor = prob > 70 ? '#ff4b2b' : '#00f260';
        text.innerText = prob.toFixed(1) + '% Уверенность (бот)';
        verdict.innerText = data.verdict || (prob > 70 ? 'ОБНАРУЖЕН БОТ' : 'ОБНАРУЖЕН ЧЕЛОВЕК');
        verdict.style.color = prob > 70 ? '#ff4b2b' : '#00f260';
    }

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#00d2ff';
        dropZone.style.background = 'rgba(0, 210, 255, 0.1)';
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '';
        dropZone.style.background = '';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '';
        dropZone.style.background = '';
        const files = Array.from(e.dataTransfer.files);
        handleFileSelection(files);
    });

    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', () => {
        const files = Array.from(fileInput.files);
        handleFileSelection(files);
    });

    function handleFileSelection(files) {
        const validFiles = files.filter(f =>
            f.name.toLowerCase().endsWith('.wav') ||
            f.name.toLowerCase().endsWith('.mp3') ||
            f.name.toLowerCase().endsWith('.m4a')
        );

        if (validFiles.length === 0) {
            alert('Пожалуйста, выберите аудиофайлы (.wav, .mp3, .m4a)');
            return;
        }

        selectedFiles = validFiles;
        fileCount.innerText = 'Выбрано файлов: ' + selectedFiles.length;
        scannerControls.style.display = 'flex';
        downloadCsvBtn.style.display = 'none';
        showSelectedFiles();
    }

    function showSelectedFiles() {
        const resultsList = document.getElementById('scan-results');
        resultsList.innerHTML = '';
        selectedFiles.forEach((file) => {
            const item = document.createElement('div');
            item.className = 'activity-item glass';
            item.style.padding = '10px';
            item.style.marginTop = '5px';
            item.innerHTML = `
                <i class="fas fa-file-audio" style="color: #00d2ff"></i>
                <div class="details">
                    <span class="title">${file.name}</span>
                    <span class="time">${formatFileSize(file.size)}</span>
                </div>
            `;
            resultsList.appendChild(item);
        });
    }

    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' байт';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' КБ';
        return (bytes / (1024 * 1024)).toFixed(1) + ' МБ';
    }

    // Model Comparison functionality
    const compareAnalyzeBtn = document.getElementById('compare-analyze-btn');
    const compareFileList = document.getElementById('compare-file-list');
    
    if (compareDropZone) {
        compareDropZone.addEventListener('click', () => compareFileInput.click());
        compareDropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            compareDropZone.style.borderColor = '#00d2ff';
            compareDropZone.style.background = 'rgba(0, 210, 255, 0.1)';
        });
        compareDropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            compareDropZone.style.borderColor = '';
            compareDropZone.style.background = '';
        });
        compareDropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            compareDropZone.style.borderColor = '';
            compareDropZone.style.background = '';
            const files = Array.from(e.dataTransfer.files);
            handleCompareFileSelection(files);
        });
        compareFileInput.addEventListener('change', () => {
            const files = Array.from(compareFileInput.files);
            handleCompareFileSelection(files);
        });
    }
    
    if (compareAnalyzeBtn) {
        compareAnalyzeBtn.addEventListener('click', analyzeCompareFiles);
    }
    
    function handleCompareFileSelection(files) {
        const validFiles = files.filter(f =>
            f.name.toLowerCase().endsWith('.wav') ||
            f.name.toLowerCase().endsWith('.mp3') ||
            f.name.toLowerCase().endsWith('.m4a')
        );
        
        if (validFiles.length === 0) {
            alert('Пожалуйста, выберите аудиофайлы (.wav, .mp3, .m4a)');
            return;
        }
        
        compareFiles = validFiles;
        showCompareFileList();
        compareAnalyzeBtn.style.display = 'inline-block';
    }
    
    function showCompareFileList() {
        compareFileList.style.display = 'block';
        compareFileList.innerHTML = '';
        
        compareFiles.forEach((file, index) => {
            const item = document.createElement('div');
            item.className = 'activity-item';
            item.style.padding = '8px';
            item.style.marginTop = '5px';
            item.innerHTML = `
                <i class="fas fa-file-audio" style="color: #00d2ff"></i>
                <span>${file.name}</span>
                <span style="color: #8b95a1; margin-left: 10px;">${formatFileSize(file.size)}</span>
                <button class="remove-file" data-index="${index}" style="margin-left: auto; background: none; border: none; color: #ff4b2b; cursor: pointer;">
                    <i class="fas fa-times"></i>
                </button>
            `;
            compareFileList.appendChild(item);
        });
        
        document.querySelectorAll('.remove-file').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const index = parseInt(e.currentTarget.getAttribute('data-index'));
                compareFiles.splice(index, 1);
                showCompareFileList();
                if (compareFiles.length === 0) {
                    compareAnalyzeBtn.style.display = 'none';
                }
            });
        });
    }
    
    async function analyzeCompareFiles() {
        if (compareFiles.length === 0) return;
        
        const recordType = document.querySelector('input[name="record-type"]:checked').value;
        
        compareAnalyzeBtn.disabled = true;
        compareAnalyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Анализ...';
        
        const allResults = [];
        
        for (let i = 0; i < compareFiles.length; i++) {
            const file = compareFiles[i];
            const formData = new FormData();
            formData.append('audio_file', file);
            
            try {
                const response = await fetch('/compare_models', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                allResults.push({
                    filename: file.name,
                    results: data.results
                });
            } catch (e) {
                console.error(`Ошибка при анализе ${file.name}:`, e);
                allResults.push({
                    filename: file.name,
                    results: {
                        mfcc40: { verdict: 'ОШИБКА', probability: 0, is_bot: null },
                        mfcc80: { verdict: 'ОШИБКА', probability: 0, is_bot: null },
                        mfcc120: { verdict: 'ОШИБКА', probability: 0, is_bot: null },
                        mfcc176: { verdict: 'ОШИБКА', probability: 0, is_bot: null }
                    }
                });
            }
        }
        
        displayCompareResults(allResults, recordType);
        
        compareAnalyzeBtn.disabled = false;
        compareAnalyzeBtn.innerHTML = '<i class="fas fa-chart-bar"></i> Сравнить модели';
    }
    
    function displayCompareResults(allResults, recordType) {
        document.getElementById('compare-results').style.display = 'block';
        
        const accuracyGrid = document.getElementById('accuracy-grid');
        accuracyGrid.innerHTML = '';
        
        const models = ['mfcc40', 'mfcc80', 'mfcc120', 'mfcc176'];
        const modelNames = { mfcc40: 'MFCC 40', mfcc80: 'MFCC 80', mfcc120: 'MFCC 120', mfcc176: 'MFCC 176' };
        
        const expectedBot = recordType === 'fake';
        
        models.forEach(model => {
            let correct = 0;
            const total = allResults.length;
            
            allResults.forEach(fileResult => {
                const result = fileResult.results[model];
                if (result.is_bot !== null) {
                    const isCorrect = (result.is_bot === expectedBot);
                    if (isCorrect) correct++;
                }
            });
            
            const accuracy = total > 0 ? ((correct / total) * 100) : 0;
            const accuracyColor = accuracy >= 80 ? '#00f260' : accuracy >= 60 ? '#ffa726' : '#ff4b2b';
            
            const card = document.createElement('div');
            card.className = 'stat-card glass';
            card.style.padding = '20px';
            card.innerHTML = `
                <span class="label">${modelNames[model]}</span>
                <span class="value" style="color: ${accuracyColor}">${accuracy.toFixed(1)}%</span>
                <span class="hint" style="margin-top: 5px;">Правильно: ${correct}/${total}</span>
            `;
            accuracyGrid.appendChild(card);
        });
        
        const tableBody = document.getElementById('compare-files-body');
        tableBody.innerHTML = '';
        
        allResults.forEach(fileResult => {
            const row = document.createElement('tr');
            
            let cellsHtml = `<td><strong>${fileResult.filename}</strong></td>`;
            
            models.forEach(model => {
                const result = fileResult.results[model];
                let verdictClass = '';
                let verdictText = 'N/A';
                
                if (result.verdict === 'БОТ') {
                    verdictClass = 'bot';
                    verdictText = `БОТ<br><span style="font-size:0.8em">${(result.probability * 100).toFixed(1)}%</span>`;
                } else if (result.verdict === 'ЧЕЛОВЕК') {
                    verdictClass = 'human';
                    verdictText = `ЧЕЛОВЕК<br><span style="font-size:0.8em">${((1 - result.probability) * 100).toFixed(1)}%</span>`;
                } else {
                    verdictClass = 'error';
                    verdictText = 'ОШИБКА';
                }
                
                cellsHtml += `<td class="${verdictClass}">${verdictText}</td>`;
            });
            
            row.innerHTML = cellsHtml;
            tableBody.appendChild(row);
        });
    }

    analyzeBtn.addEventListener('click', async () => {
        if (selectedFiles.length === 0) return;

        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Анализ...';

        scanResults = [];
        const resultsList = document.getElementById('scan-results');
        resultsList.innerHTML = '';

        const formData = new FormData();
        selectedFiles.forEach(file => {
            formData.append('audio_files', file);
        });

        try {
            const response = await fetch('/predict_batch', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            scanResults = data.results;

            scanResults.forEach(result => {
                showBatchResult(result);
            });

            downloadCsvBtn.style.display = 'inline-block';
        } catch (e) {
            console.error('Ошибка пакетной загрузки:', e);
            alert('Ошибка при анализе файлов');
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Анализировать все файлы';
        }
    });

    function showBatchResult(result) {
        const resultsList = document.getElementById('scan-results');
        let color, icon, verdict;

        if (result.error) {
            color = '#ff4b2b';
            icon = 'fa-exclamation-triangle';
            verdict = 'Ошибка: ' + result.error;
        } else {
            color = result.is_bot ? '#ff4b2b' : '#00f260';
            icon = result.is_bot ? 'fa-robot' : 'fa-user';
            verdict = result.is_bot ? 'БОТ' : 'ЧЕЛОВЕК';
        }

        const item = document.createElement('div');
        item.className = 'activity-item glass';
        item.style.padding = '15px';
        item.style.marginTop = '10px';
        item.innerHTML = `
            <i class="fas ${icon}" style="color: ${color}; font-size: 1.5em;"></i>
            <div class="details">
                <span class="title">${result.filename} — ${verdict}</span>
                <span class="time">
                    ${result.probability !== undefined ? 'Уверенность: ' + (result.probability * 100).toFixed(1) + '%' : ''}
                    ${result.latency_ms !== undefined ? ' | Задержка: ' + result.latency_ms + ' мс' : ''}
                </span>
            </div>
        `;
        resultsList.prepend(item);
    }

    downloadCsvBtn.addEventListener('click', async () => {
        if (scanResults.length === 0) return;
        const resultsJson = encodeURIComponent(JSON.stringify(scanResults));
        window.location.href = '/download_csv?results_json=' + resultsJson;
    });
});