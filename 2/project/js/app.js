document.addEventListener('DOMContentLoaded', () => {

    // ═══════════════════════════════════════════════════════════════════
    //  CONFIGURATION
    // ═══════════════════════════════════════════════════════════════════
    const API_BASE_URL = '/api/songs';
    const EEG_WS_URL = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.hostname || 'localhost'}:8765`;
    const WS_RECONNECT_DELAY_MS = 3000;

    // Auto-play: wait this many ms after first detecting an emotion before
    // starting music (gives the user a moment to see what was detected).
    const AUTOPLAY_DELAY_MS = 3000;

    // Stability: keep a rolling window of the last N emotion readings.
    // Only switch emotion if the new one wins a majority of that window.
    const HISTORY_SIZE = 10;
    const MAJORITY_THRESHOLD = 6;   // out of 10 must agree to switch

    // Minimum confidence to count a reading at all
    const MIN_CONFIDENCE = 0.62;

    // After switching emotion, lock for this long before allowing another switch
    const SWITCH_LOCK_MS = 10000;

    // ── Emotion presets ───────────────────────────────────────────────
    const EMOTION_PRESETS = {
        calm: { color: '#4fc3f7', intensity: 65 },
        happy: { color: '#ffd54f', intensity: 82 },
        angry: { color: '#ef5350', intensity: 91 },
        sad: { color: '#7e57c2', intensity: 58 },
        focused: { color: '#66bb6a', intensity: 75 },
    };
    const SUPPORTED_EMOTIONS = Object.keys(EMOTION_PRESETS);

    // ═══════════════════════════════════════════════════════════════════
    //  STATE
    // ═══════════════════════════════════════════════════════════════════
    let songCache = { calm: [], happy: [], angry: [], sad: [], focused: [] };
    let currentEmotion = null;
    let currentTrackFile = null;
    let isPlaying = false;
    let audioElement = null;
    let audioContext = null;
    let analyser = null;
    let audioSource = null;
    let animationId = null;
    let progressInterval = null;
    let simPhase = 0;
    let emotionSocket = null;
    let reconnectTimer = null;
    let socketConnected = false;
    let currentVisMode = 'bar';
    let barCtx = null;
    let radialCtx = null;
    let globeRotation = 0;
    let spikePhase = 0;

    // Stability tracking
    let emotionHistory = [];
    let switchLockedUntil = 0;
    let autoPlayTimer = null;
    let pendingEmotion = null;

    // Last received features and confidence
    let liveFeatures = null;
    let liveConfidence = null;

    // ═══════════════════════════════════════════════════════════════════
    //  DOM
    // ═══════════════════════════════════════════════════════════════════
    const flash = document.getElementById('flash');
    const root = document.documentElement;
    const appTitle = document.getElementById('appTitle');
    const eegDot = document.getElementById('eegDot');
    const eegStatus = document.getElementById('eegStatus');
    const wsStatusSpan = document.getElementById('wsStatus');
    const modeNote = document.getElementById('modeNote');
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
    const toggleBtn = document.getElementById('toggleVisBtn');
    const barCanvas = document.getElementById('visBar');
    const radialCanvas = document.getElementById('visRadial');

    // ── Live EEG data panel ───────────────────────────────────────────
    let eegDataPanel = null;

    function buildEegDataPanel() {
        const existing = document.getElementById('eegDataPanel');
        if (existing) { eegDataPanel = existing; return; }

        eegDataPanel = document.createElement('div');
        eegDataPanel.id = 'eegDataPanel';
        eegDataPanel.style.cssText = `
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 12px;
            padding: 16px 20px;
            margin: 12px 0;
            display: none;
            font-family: monospace;
        `;
        eegDataPanel.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <span style="font-size:11px; letter-spacing:2px; color:rgba(255,255,255,0.4); text-transform:uppercase;">Live EEG Signal</span>
                <span id="liveConfBadge" style="
                    font-size:12px; font-weight:bold; padding:3px 10px;
                    border-radius:20px; background:rgba(255,255,255,0.08);
                    color:#fff; letter-spacing:1px;
                ">CONF —</span>
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px;">
                ${['alpha', 'beta', 'theta', 'gamma'].map(band => `
                <div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span style="font-size:11px; color:rgba(255,255,255,0.5); text-transform:uppercase; letter-spacing:1px;">${band}</span>
                        <span id="live_${band}_val" style="font-size:12px; font-weight:bold; color:#fff;">—</span>
                    </div>
                    <div style="background:rgba(255,255,255,0.08); border-radius:4px; height:5px; overflow:hidden;">
                        <div id="live_${band}_bar" style="height:100%; width:0%; border-radius:4px; transition:width 0.6s ease;
                            background:${{ alpha: '#66bb6a', beta: '#ffd54f', theta: '#4fc3f7', gamma: '#ef5350' }[band]};"></div>
                    </div>
                </div>`).join('')}
            </div>
            <div style="margin-top:12px; display:flex; align-items:center; gap:8px;">
                <span style="font-size:11px; color:rgba(255,255,255,0.4); text-transform:uppercase; letter-spacing:1px;">Confidence</span>
                <div style="flex:1; background:rgba(255,255,255,0.08); border-radius:4px; height:6px; overflow:hidden;">
                    <div id="liveConfBar" style="height:100%; width:0%; border-radius:4px;
                        background: linear-gradient(90deg,#ef5350,#ffd54f,#66bb6a);
                        transition:width 0.6s ease;"></div>
                </div>
                <span id="liveConfPct" style="font-size:12px; font-weight:bold; color:#fff; min-width:36px; text-align:right;">—</span>
            </div>

            <div style="margin-top:14px; padding-top:12px; border-top:1px solid rgba(255,255,255,0.07);">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:11px; color:rgba(255,255,255,0.4); text-transform:uppercase; letter-spacing:1px;">Emotion History</span>
                    <span id="historyVote" style="font-size:11px; color:rgba(255,255,255,0.5);">—</span>
                </div>
                <div id="historyDots" style="display:flex; gap:5px; margin-top:8px; flex-wrap:wrap;"></div>
            </div>
        `;

        const eegBar = document.querySelector('.eeg-bar');
        if (eegBar) eegBar.insertAdjacentElement('afterend', eegDataPanel);
        else document.querySelector('.container').appendChild(eegDataPanel);
    }

    function updateEegDataPanel(features, confidence, emotion) {
        if (!eegDataPanel) return;
        eegDataPanel.style.display = 'block';

        const preset = EMOTION_PRESETS[emotion] || EMOTION_PRESETS['calm'];
        const color = preset.color;

        ['alpha', 'beta', 'theta', 'gamma'].forEach(band => {
            const val = features && typeof features[band] === 'number' ? features[band] : 0;
            const pct = Math.round(val * 100);
            const valEl = document.getElementById(`live_${band}_val`);
            const barEl = document.getElementById(`live_${band}_bar`);
            if (valEl) valEl.textContent = val.toFixed(3);
            if (barEl) barEl.style.width = pct + '%';
        });

        const confPct = Math.round((confidence || 0) * 100);
        const confBadge = document.getElementById('liveConfBadge');
        const confBar = document.getElementById('liveConfBar');
        const confPctEl = document.getElementById('liveConfPct');
        if (confBadge) {
            confBadge.textContent = `CONF ${confPct}%`;
            confBadge.style.background = color + '33';
            confBadge.style.color = color;
        }
        if (confBar) confBar.style.width = confPct + '%';
        if (confPctEl) confPctEl.textContent = confPct + '%';

        updateHistoryDots();
    }

    function updateHistoryDots() {
        const dotsEl = document.getElementById('historyDots');
        const voteEl = document.getElementById('historyVote');
        if (!dotsEl) return;

        dotsEl.innerHTML = '';
        const counts = {};
        emotionHistory.forEach(e => { counts[e] = (counts[e] || 0) + 1; });
        const winner = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];

        emotionHistory.forEach(e => {
            const dot = document.createElement('div');
            const color = EMOTION_PRESETS[e] ? EMOTION_PRESETS[e].color : '#888';
            dot.title = e;
            dot.style.cssText = `
                width:10px; height:10px; border-radius:50%;
                background:${color}; opacity:0.85;
                transition:transform 0.2s;
            `;
            dotsEl.appendChild(dot);
        });

        for (let i = emotionHistory.length; i < HISTORY_SIZE; i++) {
            const dot = document.createElement('div');
            dot.style.cssText = `
                width:10px; height:10px; border-radius:50%;
                background:rgba(255,255,255,0.1);
            `;
            dotsEl.appendChild(dot);
        }

        if (voteEl && winner) {
            voteEl.textContent = `${winner[0]} ${winner[1]}/${HISTORY_SIZE}`;
            voteEl.style.color = EMOTION_PRESETS[winner[0]] ? EMOTION_PRESETS[winner[0]].color : '#fff';
        }
    }

    // ── Calibration bar ───────────────────────────────────────────────
    const calibBanner = document.querySelector('.connect-banner');
    let calibBarWrap = null, calibBarFill = null;

    function ensureCalibBar() {
        if (calibBarWrap) return;
        calibBarWrap = document.createElement('div');
        calibBarWrap.style.cssText = `
            width:100%; margin-top:8px; display:none;
            background:rgba(255,255,255,0.08); border-radius:4px; height:6px; overflow:hidden;
        `;
        calibBarFill = document.createElement('div');
        calibBarFill.style.cssText = `
            height:100%; width:0%; background:#4fc3f7;
            border-radius:4px; transition:width 0.5s ease;
        `;
        calibBarWrap.appendChild(calibBarFill);
        if (calibBanner) calibBanner.appendChild(calibBarWrap);
    }

    function showCalibBar(progress) {
        ensureCalibBar();
        calibBarWrap.style.display = 'block';
        calibBarFill.style.width = Math.round(progress * 100) + '%';
    }

    function hideCalibBar() {
        if (calibBarWrap) calibBarWrap.style.display = 'none';
    }

    // ═══════════════════════════════════════════════════════════════════
    //  SONG CACHE
    // ═══════════════════════════════════════════════════════════════════
    async function fetchSongsForEmotion(emotion) {
        try {
            const res = await fetch(`${API_BASE_URL}/${emotion}`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const songs = await res.json();
            songCache[emotion] = songs.filter(f => f.toLowerCase().endsWith('.mp3'));
            return songCache[emotion];
        } catch (e) {
            console.error(`Failed to fetch ${emotion}:`, e);
            songCache[emotion] = [];
            return [];
        }
    }

    async function ensureSongs(emotion) {
        if (!SUPPORTED_EMOTIONS.includes(emotion)) return [];
        if (songCache[emotion] && songCache[emotion].length > 0) return songCache[emotion];
        return fetchSongsForEmotion(emotion);
    }

    function getRandomSong(emotion) {
        const songs = songCache[emotion];
        if (!songs || songs.length === 0) return null;
        const idx = Math.floor(Math.random() * songs.length);
        const fileName = songs[idx];
        return {
            file: `music/${emotion}/${fileName}`,
            name: fileName.replace(/\.mp3$/i, '').replace(/[-_]/g, ' '),
            emoji: { calm: '🌊', happy: '☀️', angry: '🔥', sad: '🌧️', focused: '🎯' }[emotion] || '🎵',
        };
    }

    // ═══════════════════════════════════════════════════════════════════
    //  STABILITY — majority vote over rolling history
    // ═══════════════════════════════════════════════════════════════════
    function pushToHistory(emotion) {
        emotionHistory.push(emotion);
        if (emotionHistory.length > HISTORY_SIZE) emotionHistory.shift();
    }

    function getMajorityEmotion() {
        if (emotionHistory.length === 0) return null;
        const counts = {};
        emotionHistory.forEach(e => { counts[e] = (counts[e] || 0) + 1; });
        const [winner, votes] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
        return votes >= MAJORITY_THRESHOLD ? winner : null;
    }

    // ═══════════════════════════════════════════════════════════════════
    //  EMOTION HANDLING
    // ═══════════════════════════════════════════════════════════════════
    async function handleEmotionMessage(data) {
        const emotion = typeof data.emotion === 'string' ? data.emotion.toLowerCase() : '';
        const confidence = typeof data.confidence === 'number' ? data.confidence : 0;
        const features = data.features || null;

        if (!SUPPORTED_EMOTIONS.includes(emotion)) return;
        if (confidence < MIN_CONFIDENCE) return;

        liveFeatures = features;
        liveConfidence = confidence;
        updateEegDataPanel(features, confidence, emotion);

        pushToHistory(emotion);

        const majorityEmotion = getMajorityEmotion();

        if (!majorityEmotion) {
            updateEmotionColors(emotion, confidence);
            setConnectionState(
                `Reading: ${emotion} (${Math.round(confidence * 100)}%)`,
                `Analyzing patterns… <strong style="color:${EMOTION_PRESETS[emotion].color}">${emotion}</strong> detected — building consensus (${emotionHistory.length}/${HISTORY_SIZE})`,
                'LIVE'
            );
            return;
        }

        if (majorityEmotion === currentEmotion && isPlaying) {
            updateEmotionColors(majorityEmotion, confidence);
            setConnectionState(
                `Live: ${majorityEmotion} (${Math.round(confidence * 100)}%)`,
                `Live EEG active — <strong style="color:${EMOTION_PRESETS[majorityEmotion].color}">${majorityEmotion}</strong> detected consistently.`,
                'LIVE'
            );
            return;
        }

        const now = Date.now();
        if (majorityEmotion !== currentEmotion && now < switchLockedUntil) return;

        if (majorityEmotion !== pendingEmotion) {
            pendingEmotion = majorityEmotion;
            if (autoPlayTimer) clearTimeout(autoPlayTimer);

            let songs = await ensureSongs(majorityEmotion);
            if (!songs || songs.length === 0) {
                const fallback = SUPPORTED_EMOTIONS.find(e => songCache[e] && songCache[e].length > 0);
                if (!fallback) return;
                songs = songCache[fallback];
            }

            updateEmotionColors(majorityEmotion, confidence);

            // Stage the new song WITHOUT updating currentEmotion yet
            const song = getRandomSong(majorityEmotion);
            if (song) {
                currentTrackFile = song.file;
                trackName.textContent = song.name;
                albumArt.textContent = song.emoji;
                trackSub.textContent = `${majorityEmotion.charAt(0).toUpperCase() + majorityEmotion.slice(1)} · BrainDance`;
                if (audioElement) { audioElement.src = song.file; audioElement.load(); }
            }

            setConnectionState(
                `Detected: ${majorityEmotion} (${Math.round(confidence * 100)}%)`,
                `<strong style="color:${EMOTION_PRESETS[majorityEmotion].color}">${majorityEmotion.toUpperCase()}</strong> confirmed — music starts in ${AUTOPLAY_DELAY_MS / 1000}s…`,
                'LIVE'
            );

            autoPlayTimer = setTimeout(async () => {
                const stillMajority = getMajorityEmotion();
                if (stillMajority !== majorityEmotion) return;

                // NOW commit the emotion change and start the correct music
                currentEmotion = majorityEmotion;
                switchLockedUntil = Date.now() + SWITCH_LOCK_MS;
                pendingEmotion = null;
                playCurrentSong();

                setConnectionState(
                    `Live: ${majorityEmotion} (${Math.round(confidence * 100)}%)`,
                    `Live EEG active — <strong style="color:${EMOTION_PRESETS[majorityEmotion].color}">${majorityEmotion}</strong> is playing.`,
                    'LIVE'
                );
            }, AUTOPLAY_DELAY_MS);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    //  UI UPDATES
    // ═══════════════════════════════════════════════════════════════════
    function updateEmotionColors(emotion, confidence) {
        const preset = EMOTION_PRESETS[emotion];
        if (!preset) return;
        root.style.setProperty('--current', preset.color);
        appTitle.style.color = preset.color;
        appTitle.style.textShadow = `0 0 30px ${preset.color}`;
        eegDot.style.background = preset.color;
        eegStatus.style.color = preset.color;

        document.querySelectorAll('.emotion-btn').forEach(btn =>
            btn.classList.toggle('active', btn.dataset.emotion === emotion)
        );
        flash.style.opacity = '0.08';
        setTimeout(() => flash.style.opacity = '0', 150);
    }

    function setConnectionState(statusText, modeHtml, eegLabel) {
        if (wsStatusSpan && statusText) wsStatusSpan.textContent = statusText;
        if (modeNote && modeHtml) modeNote.innerHTML = modeHtml;
        if (eegStatus && eegLabel) eegStatus.textContent = eegLabel;
    }

    // ═══════════════════════════════════════════════════════════════════
    //  EMOTION GRID (manual simulation buttons)
    // ═══════════════════════════════════════════════════════════════════
    function buildEmotionGrid() {
        const emojiMap = { calm: '😌', happy: '😄', angry: '😤', sad: '😢', focused: '🧘' };
        SUPPORTED_EMOTIONS.forEach(em => {
            const btn = document.createElement('button');
            btn.className = 'emotion-btn';
            btn.dataset.emotion = em;
            btn.style.setProperty('--e-color', EMOTION_PRESETS[em].color);
            btn.innerHTML = `<span class="emoji">${emojiMap[em]}</span><span class="name">${em.charAt(0).toUpperCase() + em.slice(1)}</span>`;
            btn.addEventListener('click', async () => {
                if (autoPlayTimer) clearTimeout(autoPlayTimer);
                pendingEmotion = null;
                const songs = await ensureSongs(em);
                if (!songs || songs.length === 0) return;

                // Fully commit the emotion switch immediately on manual click
                currentEmotion = em;
                updateEmotionColors(em, null);
                setCurrentSong(em);   // sets currentTrackFile, trackName, albumArt
                playCurrentSong();

                setConnectionState(
                    `Manual: ${em}`,
                    `Simulation mode — <strong style="color:${EMOTION_PRESETS[em].color}">${em}</strong> selected manually.`,
                    'SIMULATED'
                );
            });
            emotionGrid.appendChild(btn);
        });
    }

    // ═══════════════════════════════════════════════════════════════════
    //  AUDIO
    // ═══════════════════════════════════════════════════════════════════
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
                audioSource = audioContext.createMediaElementSource(audioElement);
                audioSource.connect(analyser);
                analyser.connect(audioContext.destination);
            }
            if (audioContext.state === 'suspended') audioContext.resume();
        }, { once: true });
    }

    function setCurrentSong(emotion) {
        const song = getRandomSong(emotion);
        if (!song) return;
        currentTrackFile = song.file;
        trackName.textContent = song.name;
        albumArt.textContent = song.emoji;
        trackSub.textContent = `${emotion.charAt(0).toUpperCase() + emotion.slice(1)} · BrainDance`;
        if (audioElement) { audioElement.src = song.file; audioElement.load(); }
    }

    function playCurrentSong() {
        if (!audioElement || !currentTrackFile) return;
        if (audioElement.src !== window.location.origin + '/' + currentTrackFile) {
            audioElement.src = currentTrackFile;
            audioElement.load();
        }
        audioElement.play().catch(e => console.warn('Play blocked:', e));
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

    function togglePlay() { if (isPlaying) pauseSong(); else playCurrentSong(); }

    async function nextTrack() {
        // Always use currentEmotion so the next track is from the correct folder
        const em = currentEmotion || 'calm';
        const song = getRandomSong(em);
        if (!song) return;
        currentTrackFile = song.file;
        trackName.textContent = song.name;
        albumArt.textContent = song.emoji;
        trackSub.textContent = `${em.charAt(0).toUpperCase() + em.slice(1)} · BrainDance`;
        if (audioElement) {
            audioElement.src = song.file;
            audioElement.load();
            if (isPlaying) await audioElement.play().catch(e => console.warn(e));
        }
        progressFill.style.width = '0%';
        timeElapsed.textContent = '0:00';
        timeDuration.textContent = '0:00';
    }

    function prevTrack() { nextTrack(); }

    function startProgressUpdater() {
        if (progressInterval) clearInterval(progressInterval);
        progressInterval = setInterval(updateProgress, 200);
    }

    function updateProgress() {
        if (!audioElement || !audioElement.duration || isNaN(audioElement.duration)) return;
        const pct = (audioElement.currentTime / audioElement.duration) * 100;
        progressFill.style.width = pct + '%';
        timeElapsed.textContent = formatTime(audioElement.currentTime);
        timeDuration.textContent = formatTime(audioElement.duration);
    }

    function formatTime(sec) {
        if (isNaN(sec)) return '0:00';
        sec = Math.floor(sec);
        return `${Math.floor(sec / 60)}:${(sec % 60).toString().padStart(2, '0')}`;
    }

    function hexToRgb(hex) {
        return {
            r: parseInt(hex.slice(1, 3), 16),
            g: parseInt(hex.slice(3, 5), 16),
            b: parseInt(hex.slice(5, 7), 16),
        };
    }

    // ═══════════════════════════════════════════════════════════════════
    //  VISUALIZER
    // ═══════════════════════════════════════════════════════════════════
    function setVisualizerMode(mode) {
        currentVisMode = mode;
        if (mode === 'bar') {
            barCanvas.classList.add('active'); radialCanvas.classList.remove('active');
            if (toggleBtn) toggleBtn.textContent = 'Switch to Radial View';
            requestAnimationFrame(() => { resizeBarCanvas(); barCtx = barCanvas.getContext('2d'); });
        } else {
            barCanvas.classList.remove('active'); radialCanvas.classList.add('active');
            if (toggleBtn) toggleBtn.textContent = 'Switch to Bar View';
            resizeRadialCanvas(); radialCtx = radialCanvas.getContext('2d');
        }
    }

    function resizeBarCanvas() {
        if (!barCanvas) return;
        const w = barCanvas.offsetWidth || barCanvas.parentElement.clientWidth;
        if (w > 0) { barCanvas.width = w; barCanvas.height = 300; barCtx = barCanvas.getContext('2d'); }
    }

    function resizeRadialCanvas() {
        if (!radialCanvas) return;
        radialCanvas.width = 560; radialCanvas.height = 560;
        radialCtx = radialCanvas.getContext('2d');
    }

    function drawCenteredBars(dataArray, width, height, ctx, colorRgb) {
        const bufLen = dataArray.length;
        const barWidth = width / (bufLen * 2);
        const centerX = width / 2;
        for (let i = 0; i < bufLen; i++) {
            const value = dataArray[i] / 255;
            const barHeight = Math.max(2, value * height * 0.7);
            ctx.fillStyle = `rgba(${colorRgb.r},${colorRgb.g},${colorRgb.b},${0.5 + value * 0.5})`;
            ctx.fillRect(centerX - (i + 1) * barWidth, height - barHeight, barWidth - 1, barHeight);
            ctx.fillRect(centerX + i * barWidth, height - barHeight, barWidth - 1, barHeight);
        }
    }

    function drawRadialSpectrum(dataArray, ctx, width, height, colorRgb) {
        const cx = width / 2, cy = height / 2;
        const maxR = Math.min(width, height) / 2 - 16;
        const bufLen = dataArray.length;
        const c = `${colorRgb.r},${colorRgb.g},${colorRgb.b}`;
        const avgVal = dataArray.reduce((s, v) => s + v, 0) / bufLen / 255;
        ctx.shadowBlur = 0; ctx.shadowColor = 'transparent';
        ctx.clearRect(0, 0, width, height);
        ctx.save();
        ctx.beginPath(); ctx.arc(cx, cy, maxR, 0, Math.PI * 2); ctx.clip();
        ctx.fillStyle = 'rgba(6,6,16,1)'; ctx.fillRect(0, 0, width, height);
        for (let l = 1; l < 7; l++) {
            const t = l / 7, lat = (t - 0.5) * Math.PI, cosLat = Math.cos(lat);
            const projY = cy + maxR * Math.sin(lat), projRx = maxR * cosLat, projRy = projRx * 0.18;
            ctx.beginPath(); ctx.ellipse(cx, projY, projRx, projRy, 0, 0, Math.PI * 2);
            ctx.strokeStyle = `rgba(${c},${0.06 + cosLat * 0.10})`; ctx.lineWidth = 0.75; ctx.stroke();
        }
        for (let l = 0; l < 8; l++) {
            const angle = (l / 8) * Math.PI + globeRotation, cosA = Math.cos(angle);
            ctx.beginPath(); ctx.ellipse(cx, cy, maxR * Math.abs(cosA), maxR, 0, 0, Math.PI * 2);
            ctx.strokeStyle = `rgba(${c},${0.05 + Math.abs(cosA) * 0.13})`; ctx.lineWidth = 0.75; ctx.stroke();
        }
        ctx.beginPath(); ctx.ellipse(cx, cy, maxR, maxR * 0.18, 0, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(${c},0.22)`; ctx.lineWidth = 1; ctx.stroke();
        const spikeCount = 128, half = spikeCount / 2;
        const equatorRy = maxR * 0.32, baseR = maxR * 0.72, maxSpike = maxR * 0.22, breathAmp = maxR * 0.03;
        const usedBins = Math.ceil(bufLen * 0.55);
        for (let i = 0; i < spikeCount; i++) {
            const halfIdx = i <= half ? i : spikeCount - i;
            const dataIdx = Math.min(Math.floor((halfIdx / half) * usedBins), usedBins - 1);
            const value = Math.pow(dataArray[dataIdx] / 255, 0.45);
            const angle = (i / spikeCount) * Math.PI * 2 + globeRotation;
            const cosA = Math.cos(angle), sinA = Math.sin(angle);
            const depth3d = 0.4 + (cosA + 1) * 0.3;
            const baseX = cx + baseR * cosA, baseY = cy + baseR * sinA * (equatorRy / maxR);
            const nx = cosA, ny = sinA * (equatorRy / maxR);
            const nLen = Math.sqrt(nx * nx + ny * ny) || 1;
            const spikeLen = (0.08 + value * 0.92) * maxSpike * depth3d;
            const tipX = baseX + (nx / nLen) * spikeLen, tipY = baseY + (ny / nLen) * spikeLen;
            const waveOffset = (halfIdx / half) * Math.PI * 2 * 0.6;
            const breathShift = Math.sin(spikePhase + waveOffset) * breathAmp;
            ctx.beginPath(); ctx.moveTo(baseX, baseY + breathShift * 0.3); ctx.lineTo(tipX, tipY + breathShift);
            ctx.lineWidth = (2.0 + value * 5.0) * depth3d;
            ctx.strokeStyle = `rgba(${c},${0.65 + value * 0.35})`;
            ctx.shadowColor = `rgb(${c})`; ctx.shadowBlur = value > 0.5 ? 10 : 4;
            ctx.stroke(); ctx.shadowBlur = 0; ctx.shadowColor = 'transparent';
            if (value > 0.2) {
                ctx.beginPath(); ctx.arc(tipX, tipY + breathShift, 1.5 + value * 2.5 * depth3d, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(255,255,255,${value * depth3d * 0.95})`; ctx.fill();
            }
        }
        const vigGrad = ctx.createRadialGradient(cx, cy, maxR * 0.45, cx, cy, maxR);
        vigGrad.addColorStop(0, 'rgba(0,0,0,0)'); vigGrad.addColorStop(1, 'rgba(0,0,0,0.60)');
        ctx.fillStyle = vigGrad; ctx.fillRect(0, 0, width, height);
        const coreSize = maxR * (0.055 + avgVal * 0.075);
        const coreGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreSize);
        coreGrad.addColorStop(0, `rgba(255,255,255,${0.45 + avgVal * 0.45})`);
        coreGrad.addColorStop(0.5, `rgba(${c},${0.2 + avgVal * 0.25})`);
        coreGrad.addColorStop(1, `rgba(${c},0)`);
        ctx.fillStyle = coreGrad; ctx.beginPath(); ctx.arc(cx, cy, coreSize, 0, Math.PI * 2); ctx.fill();
        ctx.restore();
        ctx.beginPath(); ctx.arc(cx, cy, maxR, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(${c},0.55)`; ctx.lineWidth = 2.5;
        ctx.shadowColor = `rgb(${c})`; ctx.shadowBlur = 20; ctx.stroke();
        ctx.shadowBlur = 0; ctx.shadowColor = 'transparent';
        globeRotation += 0.007; if (globeRotation > Math.PI * 2) globeRotation -= Math.PI * 2;
        spikePhase += 0.045; if (spikePhase > Math.PI * 200) spikePhase -= Math.PI * 200;
    }

    function getSimulatedData(length) {
        const arr = new Uint8Array(length);
        simPhase += 0.05;
        for (let i = 0; i < length; i++)
            arr[i] = Math.max(0, Math.min(255, 80 + 60 * Math.sin(simPhase + i * 0.3) + 50 * Math.sin(simPhase * 2.3 + i * 0.7)));
        return arr;
    }

    function startVisualizerLoop() {
        if (animationId) cancelAnimationFrame(animationId);
        function animate() {
            animationId = requestAnimationFrame(animate);
            const rgb = hexToRgb(EMOTION_PRESETS[currentEmotion || 'calm'].color);
            const data = analyser
                ? (() => { const b = new Uint8Array(analyser.frequencyBinCount); analyser.getByteFrequencyData(b); return b; })()
                : getSimulatedData(64);
            if (currentVisMode === 'bar' && barCtx && barCanvas && barCanvas.width > 0) {
                barCtx.clearRect(0, 0, barCanvas.width, barCanvas.height);
                drawCenteredBars(data, barCanvas.width, barCanvas.height, barCtx, rgb);
            } else if (currentVisMode === 'radial' && radialCtx && radialCanvas && radialCanvas.width > 0) {
                drawRadialSpectrum(data, radialCtx, radialCanvas.width, radialCanvas.height, rgb);
            }
        }
        animate();
    }

    // ═══════════════════════════════════════════════════════════════════
    //  WEBSOCKET
    // ═══════════════════════════════════════════════════════════════════
    function handleWsMessage(data) {
        if (!data || typeof data !== 'object') return;

        if (data.type === 'status' && data.status === 'calibrating') {
            const pct = Math.round((data.progress || 0) * 100);
            showCalibBar(data.progress || 0);
            if ((data.progress || 0) >= 1.0) {
                hideCalibBar();
                setConnectionState(
                    'Calibration complete',
                    'Baseline recorded. Analyzing your brainwaves…',
                    'LIVE'
                );
            } else {
                setConnectionState(
                    'Calibrating baseline…',
                    `EEG connected. Collecting your personal baseline… <strong>${pct}%</strong>`,
                    'CALIBRATING'
                );
            }
            return;
        }

        if (data.type === 'eeg_window') return;

        if (data.type === 'emotion') {
            hideCalibBar();
            handleEmotionMessage(data);
        }
    }

    function scheduleReconnect() {
        if (reconnectTimer) return;
        reconnectTimer = setTimeout(() => { reconnectTimer = null; connectEmotionSocket(); }, WS_RECONNECT_DELAY_MS);
    }

    function connectEmotionSocket(force = false) {
        if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
        if (emotionSocket && (emotionSocket.readyState === WebSocket.OPEN || emotionSocket.readyState === WebSocket.CONNECTING)) {
            if (!force) return;
            emotionSocket.onclose = null; emotionSocket.close();
            emotionSocket = null; socketConnected = false;
        }
        setConnectionState('Connecting…', 'Trying to connect to the live EEG stream…', 'CONNECTING');
        hideCalibBar();
        emotionSocket = new WebSocket(EEG_WS_URL);

        emotionSocket.onopen = () => {
            socketConnected = true;
            setConnectionState('EEG stream connected', 'Live EEG connected. Waiting for calibration…', 'LIVE');
        };
        emotionSocket.onmessage = (event) => {
            let data;
            try { data = JSON.parse(event.data); } catch { return; }
            handleWsMessage(data);
        };
        emotionSocket.onerror = () => {
            if (!socketConnected)
                setConnectionState('EEG unavailable',
                    'Running in <span style="color:#ffd54f">simulation mode</span>. Click emotions below.',
                    'SIMULATED');
        };
        emotionSocket.onclose = () => {
            emotionSocket = null; socketConnected = false;
            hideCalibBar();
            setConnectionState('Disconnected',
                'Running in <span style="color:#ffd54f">simulation mode</span>. Click emotions below.',
                'SIMULATED');
            scheduleReconnect();
        };
    }

    // ═══════════════════════════════════════════════════════════════════
    //  INIT
    // ═══════════════════════════════════════════════════════════════════
    async function init() {
        initAudio();
        buildEmotionGrid();
        buildEegDataPanel();

        setConnectionState(
            'WebSocket not connected',
            'Running in <span style="color:#ffd54f">simulation mode</span>. Click emotions below.',
            'SIMULATED'
        );

        playBtn.addEventListener('click', togglePlay);
        nextBtn.addEventListener('click', nextTrack);
        prevBtn.addEventListener('click', prevTrack);
        if (rewindBtn) rewindBtn.addEventListener('click', () => { if (audioElement) audioElement.currentTime -= 10; });
        if (forwardBtn) forwardBtn.addEventListener('click', () => { if (audioElement) audioElement.currentTime += 10; });
        if (reconnectBtn) reconnectBtn.addEventListener('click', () => connectEmotionSocket(true));
        if (toggleBtn) toggleBtn.addEventListener('click', () =>
            setVisualizerMode(currentVisMode === 'bar' ? 'radial' : 'bar'));
        window.addEventListener('resize', () => {
            if (currentVisMode === 'bar') resizeBarCanvas(); else resizeRadialCanvas();
        });

        await Promise.all(SUPPORTED_EMOTIONS.map(em => fetchSongsForEmotion(em)));

        updateEmotionColors('calm', null);
        setVisualizerMode('bar');
        startVisualizerLoop();
        connectEmotionSocket();
    }

    init();
});