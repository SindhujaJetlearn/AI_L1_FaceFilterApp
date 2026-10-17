/* ══════════════════════════════════════════════════════════════
   FaceFilter AI — app.js
   Handles: filter switching, keyboard shortcuts,
            photo capture, MediaRecorder video recording, gallery
   ══════════════════════════════════════════════════════════════ */

'use strict';

// ── DOM references ────────────────────────────────────────────
const filterGrid    = document.getElementById('filterGrid');
const captureBtn    = document.getElementById('captureBtn');
const recordBtn     = document.getElementById('recordBtn');
const recordLabel   = document.getElementById('recordLabel');
const resetBtn      = document.getElementById('resetBtn');
const recBadge      = document.getElementById('recBadge');
const activeLabel   = document.getElementById('activeFilterLabel');
const galleryStrip  = document.getElementById('galleryStrip');
const gallerySection= document.getElementById('gallerySection');
const toast         = document.getElementById('toast');
const videoFeed     = document.getElementById('videoFeed');

// ── State ─────────────────────────────────────────────────────
let currentFilter   = 'none';
let isRecording     = false;
let mediaRecorder   = null;
let recordedChunks  = [];
let toastTimer      = null;
let galleryCount    = 0;

// ── Filter name → display label ───────────────────────────────
const FILTER_LABELS = {
  none:       'No Filter',
  dog:        '🐶 Dog',
  sunglasses: '🕶️ Sunglasses',
  crown:      '👑 Crown',
  rainbow:    '🌈 Rainbow',
  neon:       '⚡ Neon',
  vintage:    '📷 Vintage',
  cyberpunk:  '🤖 Cyberpunk',
  sketch:     '✏️ Sketch',
  pixelate:   '🎮 Pixelate',
  mirror:     '🪞 Mirror',
};

// ── Keyboard → filter mapping ─────────────────────────────────
const KEY_FILTER_MAP = {
  '1': 'dog',
  '2': 'sunglasses',
  '3': 'crown',
  '4': 'rainbow',
  '5': 'neon',
  '6': 'vintage',
  '7': 'cyberpunk',
  '8': 'sketch',
  '9': 'pixelate',
  '0': 'mirror',
};

// ── Toast notification ────────────────────────────────────────
function showToast(msg, duration = 2200) {
  clearTimeout(toastTimer);
  toast.textContent = msg;
  toast.classList.add('show');
  toastTimer = setTimeout(() => toast.classList.remove('show'), duration);
}

// ── Set active filter ─────────────────────────────────────────
async function setFilter(filterName) {
  if (filterName === currentFilter) return;

  currentFilter = filterName;

  // update active button
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.filter === filterName);
  });

  // update label
  activeLabel.textContent = FILTER_LABELS[filterName] ?? filterName;

  // notify server
  try {
    await fetch('/set_filter', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filter: filterName }),
    });
  } catch (e) {
    showToast('⚠️ Could not reach server');
  }

  if (filterName !== 'none') {
    showToast(`✨ ${FILTER_LABELS[filterName]} applied`);
  } else {
    showToast('Filter removed');
  }
}

// ── Photo capture (server-side PNG) ──────────────────────────
async function capturePhoto() {
  try {
    const res  = await fetch('/snapshot');
    if (!res.ok) { showToast('⚠️ Capture failed'); return; }
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);

    // trigger download
    const a = document.createElement('a');
    a.href = url;
    a.download = `facefilter_${Date.now()}.png`;
    a.click();

    // add to gallery
    addToGallery(url, a.download);

    showToast('📸 Photo captured!');
  } catch (e) {
    showToast('⚠️ Capture error');
  }
}

// ── Gallery ───────────────────────────────────────────────────
function addToGallery(url, filename) {
  galleryCount++;
  gallerySection.classList.add('has-items');

  const thumb = document.createElement('div');
  thumb.className = 'gallery-thumb';
  thumb.setAttribute('role', 'img');
  thumb.setAttribute('aria-label', `Captured photo ${galleryCount}`);
  thumb.innerHTML = `
    <img src="${url}" alt="Captured photo ${galleryCount}" loading="lazy" />
    <div class="thumb-dl" title="Download">⬇️</div>
  `;
  thumb.addEventListener('click', () => {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
  });
  galleryStrip.prepend(thumb);

  // animate in
  thumb.style.opacity = '0';
  thumb.style.transform = 'scale(0.8)';
  requestAnimationFrame(() => {
    thumb.style.transition = 'opacity 0.35s ease, transform 0.35s ease';
    thumb.style.opacity = '1';
    thumb.style.transform = 'scale(1)';
  });
}

// ── Video recording (MediaRecorder on MJPEG stream) ───────────
//   We capture the <img> tag's displayed frames onto a canvas,
//   then record the canvas stream — this is the reliable approach
//   for MJPEG-backed img elements in all browsers.

let recordCanvas   = null;
let recordCtx      = null;
let recordRafId    = null;

function buildRecordCanvas() {
  recordCanvas = document.createElement('canvas');
  recordCanvas.width  = videoFeed.naturalWidth  || 640;
  recordCanvas.height = videoFeed.naturalHeight || 480;
  recordCtx = recordCanvas.getContext('2d');
}

function drawFrameToCanvas() {
  if (!isRecording) return;
  try {
    recordCtx.drawImage(videoFeed, 0, 0, recordCanvas.width, recordCanvas.height);
  } catch (_) {}
  recordRafId = requestAnimationFrame(drawFrameToCanvas);
}

function startRecording() {
  buildRecordCanvas();
  recordedChunks = [];

  const stream = recordCanvas.captureStream(30);

  // pick best supported MIME type
  const mimeType = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm']
    .find(m => MediaRecorder.isTypeSupported(m)) || 'video/webm';

  try {
    mediaRecorder = new MediaRecorder(stream, { mimeType });
  } catch (e) {
    showToast('⚠️ Recording not supported in this browser');
    return;
  }

  mediaRecorder.ondataavailable = e => {
    if (e.data && e.data.size > 0) recordedChunks.push(e.data);
  };

  mediaRecorder.onstop = () => {
    const blob = new Blob(recordedChunks, { type: mimeType });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url;
    a.download = `facefilter_rec_${Date.now()}.webm`;
    a.click();
    showToast('🎬 Video saved!');
    recordedChunks = [];
  };

  mediaRecorder.start(100); // collect every 100 ms
  isRecording = true;
  drawFrameToCanvas();

  // UI update
  recBadge.hidden = false;
  recordBtn.classList.add('recording');
  recordLabel.textContent = 'Stop';
  document.getElementById('recIcon').setAttribute('fill', 'currentColor');

  showToast('🔴 Recording started');
}

function stopRecording() {
  if (!mediaRecorder || mediaRecorder.state === 'inactive') return;
  cancelAnimationFrame(recordRafId);
  mediaRecorder.stop();
  isRecording = false;

  // UI update
  recBadge.hidden = true;
  recordBtn.classList.remove('recording');
  recordLabel.textContent = 'Record';
}

function toggleRecording() {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
}

// ── Event listeners ───────────────────────────────────────────

// Filter buttons
filterGrid.addEventListener('click', e => {
  const btn = e.target.closest('.filter-btn');
  if (!btn) return;
  setFilter(btn.dataset.filter);
});

// Action buttons
captureBtn.addEventListener('click', capturePhoto);
recordBtn.addEventListener('click', toggleRecording);
resetBtn.addEventListener('click', () => setFilter('none'));

// Keyboard shortcuts
document.addEventListener('keydown', e => {
  // ignore if user is typing in an input
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

  const key = e.key;

  if (KEY_FILTER_MAP[key]) {
    e.preventDefault();
    setFilter(KEY_FILTER_MAP[key]);
    return;
  }

  switch (key) {
    case 'Escape':
      e.preventDefault();
      setFilter('none');
      break;
    case ' ':
      e.preventDefault();
      capturePhoto();
      break;
    case 'r':
    case 'R':
      e.preventDefault();
      toggleRecording();
      break;
  }
});

// ── Video feed loaded hint ────────────────────────────────────
videoFeed.addEventListener('load', () => {
  // update canvas size once real dimensions are known
  if (recordCanvas) {
    recordCanvas.width  = videoFeed.naturalWidth  || 640;
    recordCanvas.height = videoFeed.naturalHeight || 480;
  }
});

videoFeed.addEventListener('error', () => {
  showToast('⚠️ Camera stream unavailable', 4000);
});

// ── Button ripple micro-animation ────────────────────────────
document.querySelectorAll('.action-btn, .filter-btn').forEach(btn => {
  btn.addEventListener('click', function(e) {
    const ripple = document.createElement('span');
    const rect   = btn.getBoundingClientRect();
    const size   = Math.max(rect.width, rect.height);
    ripple.style.cssText = `
      position:absolute;
      border-radius:50%;
      background:rgba(255,255,255,0.15);
      width:${size}px;height:${size}px;
      left:${e.clientX - rect.left - size/2}px;
      top:${e.clientY - rect.top  - size/2}px;
      transform:scale(0);
      animation:rippleAnim 0.5s linear;
      pointer-events:none;
    `;
    btn.appendChild(ripple);
    setTimeout(() => ripple.remove(), 500);
  });
});

// inject ripple keyframe
const style = document.createElement('style');
style.textContent = `@keyframes rippleAnim { to { transform:scale(2.5); opacity:0; } }`;
document.head.appendChild(style);

// ── Init ──────────────────────────────────────────────────────
showToast('👋 FaceFilter AI ready! Press 1–0 to choose a filter.');
