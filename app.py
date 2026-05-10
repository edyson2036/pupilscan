"""
PupilScan Backend — Flask + YOLOv8n-seg
Procesa video pupilar y extrae Tc, Ts, Tr
"""

import os, base64, tempfile, logging
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('pupilscan')

# ── App ───────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Modelo ────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get('MODEL_PATH', 'best.pt')
model = None

def load_model():
    global model
    if model is None:
        log.info(f'Cargando modelo: {MODEL_PATH}')
        model = YOLO(MODEL_PATH)
        log.info('Modelo cargado OK')
    return model

# ── Parámetros del pipeline ───────────────────────────────────────────────
CONF        = float(os.environ.get('CONF', '0.20'))
SMOOTH_WIN  = int(os.environ.get('SMOOTH_WIN', '9'))
T_LED_ON    = float(os.environ.get('T_LED_ON', '2.0'))
T_LED_OFF   = float(os.environ.get('T_LED_OFF', '6.0'))
PP_MORPH_K  = int(os.environ.get('PP_MORPH_K', '11'))
PP_MIN_PIX  = int(os.environ.get('PP_MIN_PIX', '200'))

# ── Rutas estáticas ───────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': MODEL_PATH})

# ── Ruta principal de análisis ────────────────────────────────────────────
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json(force=True)
        if not data or 'video' not in data:
            return jsonify({'error': 'No se recibió video'}), 400

        # Decodificar base64 → bytes
        b64 = data['video']
        if ',' in b64:
            b64 = b64.split(',', 1)[1]
        video_bytes = base64.b64decode(b64)

        # Guardar en archivo temporal
        suffix = '.webm'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        log.info(f'Video recibido: {len(video_bytes)/1024:.1f} KB → {tmp_path}')

        # Procesar
        result = process_video(tmp_path)

    except Exception as e:
        log.error(f'Error en /analyze: {e}', exc_info=True)
        result = {'error': str(e)}
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return jsonify(result)

# ── Pipeline de análisis pupilar ──────────────────────────────────────────
def process_video(video_path: str) -> dict:
    mdl = load_model()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {'error': 'No se pudo abrir el video'}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (PP_MORPH_K, PP_MORPH_K)
    )

    areas  = []
    times  = []
    confs  = []
    frame_idx = 0
    detected  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps
        area = None
        conf_val = 0.0

        try:
            results = mdl(frame, conf=CONF, verbose=False)
            if results[0].masks is not None:
                masks_data = results[0].masks.data.cpu().numpy()
                boxes_conf = results[0].boxes.conf.cpu().numpy()
                best_idx   = int(np.argmax(boxes_conf))
                conf_val   = float(boxes_conf[best_idx])

                mask = (masks_data[best_idx] * 255).astype(np.uint8)
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(c) >= PP_MIN_PIX:
                        (_, _), radius = cv2.minEnclosingCircle(c)
                        area = float(np.pi * radius ** 2)
                        detected += 1
        except Exception as e:
            log.warning(f'Frame {frame_idx} error: {e}')

        areas.append(area)
        times.append(round(t, 4))
        confs.append(conf_val)
        frame_idx += 1

    cap.release()

    if frame_idx == 0:
        return {'error': 'Video vacío o ilegible'}

    # ── Interpolación ────────────────────────────────────────────────────
    s = pd.Series(areas, dtype=float)
    s = s.interpolate(method='linear', limit_direction='both')
    s = s.fillna(method='bfill').fillna(method='ffill')

    # ── Suavizado ────────────────────────────────────────────────────────
    win = min(SMOOTH_WIN, len(s) - 1)
    if win % 2 == 0:
        win = max(win - 1, 3)
    smooth_arr = savgol_filter(s.values, win, 2) if win >= 3 else s.values

    t_arr  = np.array(times)
    pt_arr = smooth_arr.astype(float)

    # ── Índices de eventos ───────────────────────────────────────────────
    idx_on  = int(np.argmin(np.abs(t_arr - T_LED_ON)))
    idx_off = int(np.argmin(np.abs(t_arr - T_LED_OFF)))

    # t_min: mínimo dentro de [t_LED_ON, t_LED_OFF]
    seg_cont    = pt_arr[idx_on:idx_off]
    idx_min_rel = int(np.argmin(seg_cont)) if len(seg_cont) > 0 else 0
    idx_min     = idx_on + idx_min_rel
    t_min       = float(t_arr[idx_min])

    p_base = float(np.mean(pt_arr[:max(idx_on, 1)]))
    p_min  = float(pt_arr[idx_min])

    # t_fr: primer instante después de t_off donde P ≥ 95 % del basal
    thresh      = p_min + (p_base - p_min) * 0.95
    seg_rec     = pt_arr[idx_off:]
    idx_fr_rel  = next(
        (i for i, v in enumerate(seg_rec) if v >= thresh),
        len(seg_rec) - 1
    )
    idx_fr = idx_off + idx_fr_rel
    t_fr   = float(t_arr[idx_fr])

    Tc = round(t_min  - T_LED_ON,  3)
    Ts = round(T_LED_OFF - t_min,  3)
    Tr = round(t_fr   - T_LED_OFF, 3)

    det_pct      = round(detected / frame_idx * 100, 1)
    conf_mean    = round(float(np.mean([c for c in confs if c > 0])) if detected > 0 else 0, 3)

    # ── Gráfica P(t) ─────────────────────────────────────────────────────
    chart_b64 = build_chart(
        t_arr, s.values, pt_arr,
        T_LED_ON, T_LED_OFF, t_min, t_fr,
        Tc, Ts, Tr
    )

    # ── Respuesta normalizada ─────────────────────────────────────────────
    pt_min_v = float(np.min(pt_arr))
    pt_max_v = float(np.max(pt_arr))
    rng = pt_max_v - pt_min_v if pt_max_v != pt_min_v else 1.0
    pt_norm = ((pt_arr - pt_min_v) / rng).tolist()
    t_norm  = (t_arr / float(t_arr[-1])).tolist() if t_arr[-1] > 0 else t_arr.tolist()

    return {
        'Tc':           Tc,
        'Ts':           Ts,
        'Tr':           Tr,
        'p_base':       round(p_base, 1),
        'p_min':        round(p_min, 1),
        'det_pct':      det_pct,
        'conf_mean':    conf_mean,
        'total_frames': frame_idx,
        'detected':     detected,
        't':            t_arr.tolist(),
        'pt':           pt_arr.tolist(),
        'pt_raw':       s.values.tolist(),
        'pt_norm':      pt_norm,
        't_norm':       t_norm,
        't_min':        round(t_min, 3),
        't_fr':         round(t_fr, 3),
        'chart':        chart_b64,
    }

# ── Generación de gráfica ─────────────────────────────────────────────────
def build_chart(t, raw, smooth, t0, toff, tmin, tfr, Tc, Ts, Tr):
    BG   = '#090e1a'
    BG2  = '#0f1828'
    CYAN = '#00d4ff'
    GRN  = '#00ff9d'
    AMB  = '#ffb347'
    RED  = '#ff4d6d'
    DIM  = '#3d5280'

    fig, ax = plt.subplots(figsize=(9, 4), facecolor=BG)
    ax.set_facecolor(BG2)

    # Bandas de fase
    ax.axvspan(0,    t0,   alpha=0.10, color=CYAN, zorder=0)
    ax.axvspan(t0,   tmin, alpha=0.14, color=GRN,  zorder=0)
    ax.axvspan(tmin, toff, alpha=0.14, color=AMB,  zorder=0)
    ax.axvspan(toff, tfr,  alpha=0.14, color=RED,  zorder=0)

    # Curva raw (tenue)
    ax.plot(t, raw, color=CYAN, alpha=0.20, linewidth=0.8, zorder=1)

    # Curva suavizada
    ax.plot(t, smooth, color=CYAN, linewidth=2.0, zorder=2, label='P(t) suavizada')

    # Líneas de eventos
    for tv, lbl, clr in [
        (t0,   't₀ LED↑', CYAN),
        (tmin, 't_min',   GRN),
        (toff, 't_off↓',  AMB),
        (tfr,  't_fr',    RED),
    ]:
        ax.axvline(tv, color=clr, linestyle='--', linewidth=0.9, alpha=0.7, zorder=3)
        ax.text(tv + 0.05, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else smooth.min()*0.97,
                lbl, color=clr, fontsize=8, va='bottom', fontfamily='monospace')

    # Anotaciones Tc, Ts, Tr
    y_ann = smooth.min() * 0.93
    ax.annotate('', xy=(tmin, y_ann), xytext=(t0, y_ann),
                arrowprops=dict(arrowstyle='<->', color=GRN, lw=1.2))
    ax.text((t0+tmin)/2, y_ann*0.96, f'Tc={Tc:.2f}s',
            ha='center', va='top', color=GRN, fontsize=8, fontfamily='monospace')

    ax.annotate('', xy=(toff, y_ann), xytext=(tmin, y_ann),
                arrowprops=dict(arrowstyle='<->', color=AMB, lw=1.2))
    ax.text((tmin+toff)/2, y_ann*0.96, f'Ts={Ts:.2f}s',
            ha='center', va='top', color=AMB, fontsize=8, fontfamily='monospace')

    ax.annotate('', xy=(tfr, y_ann), xytext=(toff, y_ann),
                arrowprops=dict(arrowstyle='<->', color=RED, lw=1.2))
    ax.text((toff+tfr)/2, y_ann*0.96, f'Tr={Tr:.2f}s',
            ha='center', va='top', color=RED, fontsize=8, fontfamily='monospace')

    # Estilo ejes
    ax.set_xlabel('Tiempo (s)', color=DIM, fontsize=10, fontfamily='monospace')
    ax.set_ylabel('Área pupilar (px²)', color=DIM, fontsize=10, fontfamily='monospace')
    ax.tick_params(colors=DIM, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#1e2d4a')
    ax.grid(color='#1e2d4a', linewidth=0.4, alpha=0.6)

    patches = [
        mpatches.Patch(color=CYAN, alpha=0.5, label='Basal'),
        mpatches.Patch(color=GRN,  alpha=0.5, label=f'Contracción Tc={Tc:.2f}s'),
        mpatches.Patch(color=AMB,  alpha=0.5, label=f'Estática Ts={Ts:.2f}s'),
        mpatches.Patch(color=RED,  alpha=0.5, label=f'Recuperación Tr={Tr:.2f}s'),
    ]
    ax.legend(handles=patches, fontsize=8, facecolor='#0f1828',
              labelcolor='#7a90b5', loc='upper right',
              framealpha=0.8, edgecolor='#1e2d4a')

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=110, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()

# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
