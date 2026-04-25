document.addEventListener('DOMContentLoaded', () => {
    // ================= CONFIGURATION =================
    const API_BASE_URL = '/api/songs';

    const EMOTION_PRESETS = {
        calm: { color: '#4fc3f7', waves: { delta: 12, theta: 35, alpha: 80, beta: 20, gamma: 8 }, intensity: 65, visStyle: 'smooth' },
        happy: { color: '#ffd54f', waves: { delta: 5, theta: 15, alpha: 40, beta: 60, gamma: 35 }, intensity: 82, visStyle: 'bouncy' },
        angry: { color: '#ef5350', waves: { delta: 3, theta: 8, alpha: 12, beta: 85, gamma: 70 }, intensity: 91, visStyle: 'spiky' },
        sad: { color: '#7e57c2', waves: { delta: 20, theta: 45, alpha: 25, beta: 30, gamma: 10 }, intensity: 58, visStyle: 'slow' },
        focused: { color: '#66bb6a', waves: { delta: 4, theta: 20, alpha: 65, beta: 55, gamma: 30 }, intensity: 75, visStyle: 'steady' }
    };

    let songCache = { calm: [], happy: [], angry: [], sad: [], focused: [] };

    async function fetchSongsForEmotion(emotion) {
        try {
            const response = await fetch(`${API_BASE_URL}/${emotion}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const songs = await response.json();
            songCache[emotion] = songs.filter(f => f.toLowerCase().endsWith('.mp3'));
            return songCache[emotion];
        } catch (error) {
            console.error(`Failed to fetch ${emotion}:`, error);
            songCache[emotion] = [];
            return [];
        }
    }

    function getRandomSong(emotion) {
        const songs = songCache[emotion];
        if (!songs || songs.length === 0) return null;
        const randomIndex = Math.floor(Math.random() * songs.length);
        const fileName = songs[randomIndex];
        const fullPath = `music/${emotion}/${fileName}`;
        const display = fileName.replace(/\.mp3$/i, '').replace(/_/g, ' ');
        return { file: fullPath, name: display, emoji: getEmojiForEmotion(emotion) };
    }

    function getEmojiForEmotion(emotion) {
        const map = { calm: '🌊', happy: '☀️', angry: '🔥', sad: '🌧️', focused: '🎯' };
        return map[emotion] || '🎵';
    }

    // ================= DOM ELEMENTS =================
    const flash = document.getElementById('flash');
    const root = document.documentElement;
    const appTitle = document.getElementById('appTitle');
    const eegDot = document.getElementById('eegDot');
    const eegStatus = document.getElementById('eegStatus');
    const intensityVal = document.getElementById('intensityVal');
    const intensityFill = document.getElementById('intensityFill');
    const trackName = document.getElementById('trackName');
    const trackSub = document.getElementById('trackSub');
    const albumArt = document.getElementById('albumArt');
    const playingTag = document.getElementById('playingTag');
    const playBtn = document.getElementById('playBtn');
    const timeElapsed = document.getElementById('timeElapsed');
    const timeDuration = document.getElementById('timeDuration');
    const progressFill = document.getElementById('progressFill');
    const emotionGrid = document.getElementById('emotionGrid');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const rewindBtn = document.getElementById('rewindBtn');
    const forwardBtn = document.getElementById('forwardBtn');
    const reconnectBtn = document.getElementById('reconnectBtn');
    const wsStatusSpan = document.getElementById('wsStatus');
    const toggleBtn = document.getElementById('toggleVisBtn');

    // Visualizer canvases
    const barCanvas = document.getElementById('visBar');
    const radialCanvas = document.getElementById('visRadial');
    let barCtx = null;
    let radialCtx = null;
    let currentVisMode = 'bar';

    // Audio state
    let currentEmotion = 'calm';
    let currentTrackFile = null;
    let currentTrackDisplayName = '';
    let isPlaying = false;
    let audioElement = null;
    let audioContext = null;
    let analyser = null;
    let source = null;
    let animationId = null;
    let progressInterval = null;
    let simPhase = 0;

    // ================= VISUALIZER TOGGLE & SIZING =================
    function setVisualizerMode(mode) {
        currentVisMode = mode;
        if (mode === 'bar') {
            barCanvas.classList.add('active');
            radialCanvas.classList.remove('active');
            toggleBtn.textContent = 'Switch to Radial View';
            requestAnimationFrame(() => {
                resizeBarCanvas();
                barCtx = barCanvas.getContext('2d');
            });
        } else {
            barCanvas.classList.remove('active');
            radialCanvas.classList.add('active');
            toggleBtn.textContent = 'Switch to Bar View';
            resizeRadialCanvas();
            radialCtx = radialCanvas.getContext('2d');
        }
    }

    function resizeBarCanvas() {
        if (!barCanvas) return;
        const w = barCanvas.offsetWidth || barCanvas.parentElement.clientWidth;
        if (w > 0) {
            barCanvas.width = w;
            barCanvas.height = 300;
            barCtx = barCanvas.getContext('2d');
        }
    }

    function resizeRadialCanvas() {
        if (!radialCanvas) return;
        radialCanvas.width = 560;
        radialCanvas.height = 560;
        radialCtx = radialCanvas.getContext('2d');
    }

    // ================= VISUALIZER DRAWING =================
    function drawCenteredBars(dataArray, width, height, ctx, colorRgb) {
        const bufferLength = dataArray.length;
        const barWidth = width / (bufferLength * 2);
        const centerX = width / 2;
        for (let i = 0; i < bufferLength; i++) {
            const value = dataArray[i] / 255;
            const barHeight = Math.max(2, value * height * 0.7);
            const xLeft = centerX - (i + 1) * barWidth;
            const xRight = centerX + i * barWidth;
            ctx.fillStyle = `rgba(${colorRgb.r},${colorRgb.g},${colorRgb.b},${0.5 + value * 0.5})`;
            ctx.fillRect(xLeft, height - barHeight, barWidth - 1, barHeight);
            ctx.fillRect(xRight, height - barHeight, barWidth - 1, barHeight);
        }
    }

    // ── Globe state ───────────────────────────────────────────────────────────
    let globeRotation = 0;
    let spikePhase = 0; // drives the up/down breathing animation

    function drawRadialSpectrum(dataArray, ctx, width, height, colorRgb) {
        const cx = width / 2;
        const cy = height / 2;
        const maxR = Math.min(width, height) / 2 - 16;
        const bufLen = dataArray.length;
        const c = `${colorRgb.r},${colorRgb.g},${colorRgb.b}`;
        const avgVal = dataArray.reduce((s, v) => s + v, 0) / bufLen / 255;

        // ── Reset shadow & clear ──────────────────────────────────────────
        ctx.shadowBlur = 0;
        ctx.shadowColor = 'transparent';
        ctx.clearRect(0, 0, width, height);

        // ── Clip everything to circle ─────────────────────────────────────
        ctx.save();
        ctx.beginPath();
        ctx.arc(cx, cy, maxR, 0, Math.PI * 2);
        ctx.clip();

        // ── 1. Clean dark background ──────────────────────────────────────
        ctx.fillStyle = 'rgba(6,6,16,1)';
        ctx.fillRect(0, 0, width, height);

        // ── 2. Latitude lines ─────────────────────────────────────────────
        const latLines = 7;
        for (let l = 1; l < latLines; l++) {
            const t = l / latLines;
            const lat = (t - 0.5) * Math.PI;
            const cosLat = Math.cos(lat);
            const projY = cy + maxR * Math.sin(lat);
            const projRx = maxR * cosLat;
            const projRy = projRx * 0.18;
            const alpha = 0.06 + cosLat * 0.10;
            ctx.beginPath();
            ctx.ellipse(cx, projY, projRx, projRy, 0, 0, Math.PI * 2);
            ctx.strokeStyle = `rgba(${c},${alpha})`;
            ctx.lineWidth = 0.75;
            ctx.stroke();
        }

        // ── 3. Longitude lines ────────────────────────────────────────────
        const lonLines = 8;
        for (let l = 0; l < lonLines; l++) {
            const angle = (l / lonLines) * Math.PI + globeRotation;
            const cosA = Math.cos(angle);
            const facing = Math.abs(cosA);
            const alpha = 0.05 + facing * 0.13;
            ctx.beginPath();
            ctx.ellipse(cx, cy, maxR * facing, maxR, 0, 0, Math.PI * 2);
            ctx.strokeStyle = `rgba(${c},${alpha})`;
            ctx.lineWidth = 0.75;
            ctx.stroke();
        }

        // ── 4. Equator ring highlight ─────────────────────────────────────
        ctx.beginPath();
        ctx.ellipse(cx, cy, maxR, maxR * 0.18, 0, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(${c},0.22)`;
        ctx.lineWidth = 1;
        ctx.stroke();

        // ── 5. Frequency spikes — true full-circle mirror ─────────────────
        const spikeCount = 128;
        const half = spikeCount / 2;
        const equatorRy = maxR * 0.32;
        const baseR = maxR * 0.72;  // pulled in to give spikes room
        const maxSpike = maxR * 0.22; // 0.72 + 0.22 = 0.94 maxR — safely inside the circle
        const breathAmp = maxR * 0.03;
        const breathFreq = .6;
        const breathSpeed = spikePhase;

        // Trim: FFT bins above ~55% of bufLen are near-silent for music.
        // Only sample the lower range where actual audio energy lives.
        const usedBins = Math.ceil(bufLen * 0.55);

        for (let i = 0; i < spikeCount; i++) {
            // Mountain curve: low freq at 0°, rises to high at 180°, back to low at 360°.
            // Spikes at +θ and -θ from 0° share the same halfIdx → perfect left-right mirror.
            // High frequencies now sit on the true opposite side of the circle from the low.
            const halfIdx = i <= half ? i : (spikeCount - i);
            const dataIdx = Math.min(Math.floor((halfIdx / half) * usedBins), usedBins - 1);

            // Lift the floor so quiet frequencies still show
            const value = Math.pow(dataArray[dataIdx] / 255, 0.45);

            // Spike angle rotates with globe for 3D spin
            const angle = (i / spikeCount) * Math.PI * 2 + globeRotation;
            const cosA = Math.cos(angle);
            const sinA = Math.sin(angle);

            // 3D depth: front spikes full size, back spikes shrink
            const depth3d = 0.4 + (cosA + 1) * 0.3; // 0.4 → 1.0

            // Base position on equator ellipse
            const baseX = cx + baseR * cosA;
            const baseY = cy + baseR * sinA * (equatorRy / maxR);

            // Use true outward normal from the sphere centre
            const nx = cosA;
            const ny = sinA * (equatorRy / maxR);
            const nLen = Math.sqrt(nx * nx + ny * ny) || 1;

            const spikeLen = (0.08 + value * 0.92) * maxSpike * depth3d;

            // Tip = base + normalized outward * spike length
            const tipX = baseX + (nx / nLen) * spikeLen;
            const tipY = baseY + (ny / nLen) * spikeLen;

            // Breathing — keyed on halfIdx so opposite spikes ripple identically
            const waveOffset = (halfIdx / half) * Math.PI * 2 * breathFreq;
            const breathShift = Math.sin(breathSpeed + waveOffset) * breathAmp;

            const finalTipY = tipY + breathShift;
            const finalBaseY = baseY + breathShift * 0.3;

            const alpha = 0.65 + value * 0.35;
            const lineW = (2.0 + value * 5.0) * depth3d;

            ctx.beginPath();
            ctx.moveTo(baseX, finalBaseY);
            ctx.lineTo(tipX, finalTipY);
            ctx.lineWidth = lineW;
            ctx.strokeStyle = `rgba(${c},${alpha})`;
            ctx.shadowColor = `rgb(${c})`;
            ctx.shadowBlur = value > 0.5 ? 10 : 4;
            ctx.stroke();
            ctx.shadowBlur = 0;
            ctx.shadowColor = 'transparent';

            // Glowing tip dot
            if (value > 0.2) {
                ctx.beginPath();
                ctx.arc(tipX, finalTipY, 1.5 + value * 2.5 * depth3d, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(255,255,255,${value * depth3d * 0.95})`;
                ctx.fill();
            }
        }
        // ── 6. Dark vignette for 3D depth ────────────────────────────────
        const vigGrad = ctx.createRadialGradient(cx, cy, maxR * 0.45, cx, cy, maxR);
        vigGrad.addColorStop(0, 'rgba(0,0,0,0)');
        vigGrad.addColorStop(1, 'rgba(0,0,0,0.60)');
        ctx.fillStyle = vigGrad;
        ctx.fillRect(0, 0, width, height);

        // ── 7. Pulsing centre core ────────────────────────────────────────
        const coreSize = maxR * (0.055 + avgVal * 0.075);
        const coreGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreSize);
        coreGrad.addColorStop(0, `rgba(255,255,255,${0.45 + avgVal * 0.45})`);
        coreGrad.addColorStop(0.5, `rgba(${c},${0.2 + avgVal * 0.25})`);
        coreGrad.addColorStop(1, `rgba(${c},0)`);
        ctx.fillStyle = coreGrad;
        ctx.beginPath();
        ctx.arc(cx, cy, coreSize, 0, Math.PI * 2);
        ctx.fill();

        // ── End clip ──────────────────────────────────────────────────────
        ctx.restore();

        // ── 8. Outer rim glow (outside clip = always crisp) ──────────────
        ctx.beginPath();
        ctx.arc(cx, cy, maxR, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(${c},0.55)`;
        ctx.lineWidth = 2.5;
        ctx.shadowColor = `rgb(${c})`;
        ctx.shadowBlur = 20;
        ctx.stroke();
        ctx.shadowBlur = 0;
        ctx.shadowColor = 'transparent';

        // ── Advance animation state ───────────────────────────────────────
        globeRotation += 0.007;
        if (globeRotation > Math.PI * 2) globeRotation -= Math.PI * 2;

        spikePhase += 0.045; // controls breathing speed — tweak freely
        if (spikePhase > Math.PI * 200) spikePhase -= Math.PI * 200;
    }

    // ================= SIMULATED DATA (before AudioContext starts) ==========
    function getSimulatedData(length) {
        const arr = new Uint8Array(length);
        simPhase += 0.05;
        for (let i = 0; i < length; i++) {
            arr[i] = Math.max(0, Math.min(255,
                80 + 60 * Math.sin(simPhase + i * 0.3) +
                50 * Math.sin(simPhase * 2.3 + i * 0.7)
            ));
        }
        return arr;
    }

    // ================= VISUALIZER LOOP =====================================
    function startVisualizerLoop() {
        if (animationId) cancelAnimationFrame(animationId);

        function animate() {
            animationId = requestAnimationFrame(animate);
            const colorHex = EMOTION_PRESETS[currentEmotion].color;
            const rgb = hexToRgb(colorHex);

            let dataArray;
            if (analyser) {
                const buf = new Uint8Array(analyser.frequencyBinCount);
                analyser.getByteFrequencyData(buf);
                dataArray = buf;
            } else {
                dataArray = getSimulatedData(64);
            }

            if (currentVisMode === 'bar' && barCtx && barCanvas) {
                const w = barCanvas.width, h = barCanvas.height;
                if (w > 0 && h > 0) {
                    barCtx.clearRect(0, 0, w, h);
                    drawCenteredBars(dataArray, w, h, barCtx, rgb);
                }
            } else if (currentVisMode === 'radial' && radialCtx && radialCanvas) {
                const w = radialCanvas.width, h = radialCanvas.height;
                if (w > 0 && h > 0) {
                    drawRadialSpectrum(dataArray, radialCtx, w, h, rgb);
                }
            }
        }
        animate();
    }

    // ================= AUDIO SETUP =========================================
    function initAudio() {
        audioElement = new Audio();
        audioElement.crossOrigin = 'anonymous';
        audioElement.addEventListener('ended', () => nextTrack());
        audioElement.addEventListener('timeupdate', updateProgress);
        audioElement.addEventListener('loadedmetadata', () => {
            if (!isNaN(audioElement.duration))
                timeDuration.textContent = formatTime(audioElement.duration);
        });

        document.body.addEventListener('click', () => {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioContext.createAnalyser();
                analyser.fftSize = 256;
                source = audioContext.createMediaElementSource(audioElement);
                source.connect(analyser);
                analyser.connect(audioContext.destination);
            }
            if (audioContext && audioContext.state === 'suspended') audioContext.resume();
        }, { once: true });
    }

    function hexToRgb(hex) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return { r, g, b };
    }

    // ================= AUDIO PLAYBACK ======================================
    function setCurrentSong(emotion) {
        const song = getRandomSong(emotion);
        if (!song) return;
        currentTrackFile = song.file;
        currentTrackDisplayName = song.name;
        trackName.textContent = currentTrackDisplayName;
        albumArt.textContent = song.emoji;
        trackSub.textContent = `${emotion.charAt(0).toUpperCase() + emotion.slice(1)} · NeuroBeats`;
        if (audioElement) {
            audioElement.src = currentTrackFile;
            audioElement.load();
        }
    }

    function playCurrentSong() {
        if (!audioElement || !currentTrackFile) return;
        if (audioElement.src !== window.location.origin + '/' + currentTrackFile) {
            audioElement.src = currentTrackFile;
            audioElement.load();
        }
        audioElement.play().catch(e => console.warn('Play error:', e));
        isPlaying = true;
        playBtn.textContent = '⏸';
        playingTag.textContent = 'PLAYING';
        startProgressUpdater();
    }

    function pauseSong() {
        if (!audioElement) return;
        audioElement.pause();
        isPlaying = false;
        playBtn.textContent = '▶';
        playingTag.textContent = 'PAUSED';
        if (progressInterval) clearInterval(progressInterval);
    }

    function togglePlay() {
        if (isPlaying) pauseSong();
        else playCurrentSong();
    }

    async function nextTrack() {
        const song = getRandomSong(currentEmotion);
        if (!song) return;
        currentTrackFile = song.file;
        currentTrackDisplayName = song.name;
        trackName.textContent = currentTrackDisplayName;
        albumArt.textContent = song.emoji;
        if (audioElement) {
            audioElement.src = currentTrackFile;
            audioElement.load();
            if (isPlaying) await audioElement.play().catch(e => console.warn(e));
        }
    }

    function prevTrack() { nextTrack(); }

    function startProgressUpdater() {
        if (progressInterval) clearInterval(progressInterval);
        progressInterval = setInterval(() => {
            if (audioElement && audioElement.duration && !isNaN(audioElement.duration)) {
                const percent = (audioElement.currentTime / audioElement.duration) * 100;
                progressFill.style.width = percent + '%';
                timeElapsed.textContent = formatTime(audioElement.currentTime);
            }
        }, 200);
    }

    function updateProgress() {
        if (audioElement && audioElement.duration && !isNaN(audioElement.duration)) {
            const percent = (audioElement.currentTime / audioElement.duration) * 100;
            progressFill.style.width = percent + '%';
            timeElapsed.textContent = formatTime(audioElement.currentTime);
            timeDuration.textContent = formatTime(audioElement.duration);
        }
    }

    function formatTime(sec) {
        if (isNaN(sec)) return '0:00';
        sec = Math.floor(sec);
        return `${Math.floor(sec / 60)}:${(sec % 60).toString().padStart(2, '0')}`;
    }

    // ================= UI UPDATES ==========================================
    function updateEmotionUI(emotion) {
        const preset = EMOTION_PRESETS[emotion];
        if (!preset) return;
        root.style.setProperty('--current', preset.color);
        appTitle.style.color = preset.color;
        appTitle.style.textShadow = `0 0 30px ${preset.color}`;
        eegDot.style.background = preset.color;
        eegStatus.style.color = preset.color;
        intensityVal.textContent = preset.intensity + '%';
        intensityFill.style.width = preset.intensity + '%';
        animateBrainwaves(preset.waves);
        document.querySelectorAll('.emotion-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.emotion === emotion);
        });
        flash.style.opacity = '0.08';
        setTimeout(() => flash.style.opacity = '0', 150);
    }

    function animateBrainwaves(waves) {
        for (let band in waves) {
            const cap = band.charAt(0).toUpperCase() + band.slice(1);
            const valEl = document.getElementById('w' + cap);
            const barEl = document.getElementById('w' + cap + 'Bar');
            if (valEl) valEl.textContent = waves[band];
            if (barEl) barEl.style.width = Math.min(waves[band], 100) + '%';
        }
    }

    // ================= EMOTION GRID ========================================
    function buildEmotionGrid() {
        const emotions = ['calm', 'happy', 'angry', 'sad', 'focused'];
        const emojiMap = { calm: '😌', happy: '😄', angry: '😤', sad: '😢', focused: '🧘' };
        emotions.forEach(em => {
            const btn = document.createElement('button');
            btn.className = 'emotion-btn';
            btn.dataset.emotion = em;
            btn.style.setProperty('--e-color', EMOTION_PRESETS[em].color);
            btn.innerHTML = `<span class="emoji">${emojiMap[em]}</span><span class="name">${em.charAt(0).toUpperCase() + em.slice(1)}</span>`;
            btn.addEventListener('click', async () => {
                if (songCache[em].length === 0) await fetchSongsForEmotion(em);
                currentEmotion = em;
                setCurrentSong(em);
                playCurrentSong();
                updateEmotionUI(em);
            });
            emotionGrid.appendChild(btn);
        });
    }

    // ================= INITIALIZE ==========================================
    async function init() {
        initAudio();
        buildEmotionGrid();

        playBtn.addEventListener('click', togglePlay);
        nextBtn.addEventListener('click', nextTrack);
        prevBtn.addEventListener('click', prevTrack);
        if (rewindBtn) rewindBtn.addEventListener('click', () => { if (audioElement) audioElement.currentTime -= 10; });
        if (forwardBtn) forwardBtn.addEventListener('click', () => { if (audioElement) audioElement.currentTime += 10; });
        if (reconnectBtn) reconnectBtn.addEventListener('click', () => { wsStatusSpan.textContent = 'Reconnecting...'; });

        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => {
                setVisualizerMode(currentVisMode === 'bar' ? 'radial' : 'bar');
            });
        }

        window.addEventListener('resize', () => {
            if (currentVisMode === 'bar') resizeBarCanvas();
            else resizeRadialCanvas();
        });

        await Promise.all(Object.keys(songCache).map(em => fetchSongsForEmotion(em)));
        setCurrentSong('calm');
        updateEmotionUI('calm');

        setVisualizerMode('bar');
        startVisualizerLoop();
    }

    init();
});