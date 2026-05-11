import os, base64, tempfile, logging
from io import BytesIO
import cv2
import numpy as np
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import onnxruntime as ort
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('pupilscan')

app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_PATH = os.environ.get('MODEL_PATH', 'best.onnx')
INPUT_SIZE  = 640
ort_session = None

CONF       = float(os.environ.get('CONF',       '0.20'))
SMOOTH_WIN = int(os.environ.get('SMOOTH_WIN',   '9'))
T_LED_ON   = float(os.environ.get('T_LED_ON',   '2.0'))
T_LED_OFF  = float(os.environ.get('T_LED_OFF',  '6.0'))
PP_MORPH_K = int(os.environ.get('PP_MORPH_K',  '11'))
PP_MIN_PIX = int(os.environ.get('PP_MIN_PIX',  '200'))


def load_model():
    global ort_session
    if ort_session is None:
        log.info(f'Cargando modelo ONNX: {MODEL_PATH}')
        ort_session = ort.InferenceSession(
            MODEL_PATH, providers=['CPUExecutionProvider']
        )
        log.info('Modelo ONNX cargado OK')
    return ort_session


def preprocess(frame):
    h, w = frame.shape[:2]
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img, h, w


def postprocess_mask(output, orig_h, orig_w):
    try:
        det   = output[0][0].T   # (8400, 37)
        proto = output[1][0]     # (32, 160, 160)

        # Col 4 = confianza objeto
        scores = det[:, 4]
        mask_ok = scores >= CONF
        if not np.any(mask_ok):
            return None, 0.0

        det_f     = det[mask_ok]
        best_idx  = int(np.argmax(det_f[:, 4]))
        best_det  = det_f[best_idx]
        best_conf = float(best_det[4])

        # 1 clase → cols: 0-3 bbox, 4 conf, 5 cls_score, 5:37 mask coefs (32 valores)
        mask_coefs = best_det[5:37].reshape(1, 32)

        proto_flat = proto.reshape(32, -1)
        mask_flat  = (mask_coefs @ proto_flat).reshape(160, 160)
        mask_sig   = 1.0 / (1.0 + np.exp(-mask_flat))
        mask_bin   = (mask_sig > 0.5).astype(np.uint8) * 255
        mask_full  = cv2.resize(mask_bin, (orig_w, orig_h))

        return mask_full, best_conf

    except Exception as e:
        log.warning(f'postprocess: {e}')
        return None, 0.0


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': MODEL_PATH})


@app.route('/analyze', methods=['POST'])
def analyze():
    tmp_path = None
    try:
        data = request.get_json(force=True)
        if not data or 'video' not in data:
            return jsonify({'error': 'No se recibio video'}), 400

        b64 = data['video']
        if ',' in b64:
            b64 = b64.split(',', 1)[1]
        video_bytes = base64.b64decode(b64)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        log.info(f'Video recibido: {len(video_bytes)/1024:.1f} KB')
        result = process_video(tmp_path)

    except Exception as e:
        log.error(f'Error /analyze: {e}', exc_info=True)
        result = {'error': str(e)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return jsonify(result)


def process_video(video_path):
    sess = load_model()
    cap  = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {'error': 'No se pudo abrir el video'}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0

    input_name = sess.get_inputs()[0].name
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (PP_MORPH_K, PP_MORPH_K)
    )

    areas, times, conf_list = [], [], []
    frame_idx = 0
    detected  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps

        # Procesar 1 de cada 2 frames para ahorrar memoria
        if frame_idx % 4 != 0:
            areas.append(None)
            times.append(round(t, 4))
            frame_idx += 1
            continue

        area = None
        try:
            img, orig_h, orig_w = preprocess(frame)
            outputs = sess.run(None, {input_name: img})
            mask, conf_val = postprocess_mask(outputs, orig_h, orig_w)

            if mask is not None:
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
                        conf_list.append(conf_val)
        except Exception as e:
            log.warning(f'Frame {frame_idx}: {e}')

        areas.append(area)
        times.append(round(t, 4))
        frame_idx += 1

    cap.release()

    if frame_idx == 0:
        return {'error': 'Video vacio o ilegible'}

    # Interpolacion con numpy
    areas_arr = np.array(areas, dtype=float)
    nans = np.isnan(areas_arr)
    if np.all(nans):
        return {'error': 'No se detecto la pupila en ningun frame'}

    x = np.arange(len(areas_arr))
    areas_arr[nans] = np.interp(x[nans], x[~nans], areas_arr[~nans])

    # Suavizado
    win = min(SMOOTH_WIN, len(areas_arr) - 1)
    if win % 2 == 0:
        win = max(win - 1, 3)
    smooth = savgol_filter(areas_arr, win, 2) if win >= 3 else areas_arr.copy()

    t_arr  = np.array(times)
    pt_arr = smooth.astype(float)

    # Eventos temporales
    idx_on  = int(np.argmin(np.abs(t_arr - T_LED_ON)))
    idx_off = int(np.argmin(np.abs(t_arr - T_LED_OFF)))

    seg_cont    = pt_arr[idx_on:idx_off]
    idx_min_rel = int(np.argmin(seg_cont)) if len(seg_cont) > 0 else 0
    idx_min     = idx_on + idx_min_rel
    t_min       = float(t_arr[idx_min])

    p_base = float(np.mean(pt_arr[:max(idx_on, 1)]))
    p_min  = float(pt_arr[idx_min])

    thresh     = p_min + (p_base - p_min) * 0.95
    seg_rec    = pt_arr[idx_off:]
    idx_fr_rel = next(
        (i for i, v in enumerate(seg_rec) if v >= thresh),
        len(seg_rec) - 1
    )
    idx_fr = idx_off + idx_fr_rel
    t_fr   = float(t_arr[idx_fr])

    Tc = round(t_min  - T_LED_ON,  3)
    Ts = round(T_LED_OFF - t_min,  3)
    Tr = round(t_fr   - T_LED_OFF, 3)

    det_pct   = round(detected / frame_idx * 100, 1)
    conf_mean = round(float(np.mean(conf_list)) if conf_list else 0.0, 3)

    chart_b64 = build_chart(
        t_arr, areas_arr, pt_arr,
        T_LED_ON, T_LED_OFF, t_min, t_fr,
        Tc, Ts, Tr
    )

    rng = float(np.max(pt_arr) - np.min(pt_arr))
    if rng == 0:
        rng = 1.0

    return {
        'Tc': Tc,
        'Ts': Ts,
        'Tr': Tr,
        'p_base':      round(p_base, 1),
        'p_min':       round(p_min, 1),
        'det_pct':     det_pct,
        'conf_mean':   conf_mean,
        'total_frames': frame_idx,
        'detected':    detected,
        't':           t_arr.tolist(),
        'pt':          pt_arr.tolist(),
        'pt_norm':     ((pt_arr - np.min(pt_arr)) / rng).tolist(),
        't_norm':      (t_arr / float(t_arr[-1])).tolist() if t_arr[-1] > 0 else t_arr.tolist(),
        't_min':       round(t_min, 3),
        't_fr':        round(t_fr,  3),
        'chart':       chart_b64,
    }


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

    ax.axvspan(0,    t0,   alpha=0.10, color=CYAN)
    ax.axvspan(t0,   tmin, alpha=0.14, color=GRN)
    ax.axvspan(tmin, toff, alpha=0.14, color=AMB)
    ax.axvspan(toff, tfr,  alpha=0.14, color=RED)

    ax.plot(t, raw,    color=CYAN, alpha=0.20, linewidth=0.8)
    ax.plot(t, smooth, color=CYAN, linewidth=2.0)

    for tv, clr in [(t0, CYAN), (tmin, GRN), (toff, AMB), (tfr, RED)]:
        ax.axvline(tv, color=clr, linestyle='--', linewidth=0.9, alpha=0.7)

    y_ann = smooth.min() * 0.93
    for ta, tb, lbl, clr in [
        (t0,   tmin, f'Tc={Tc:.2f}s', GRN),
        (tmin, toff, f'Ts={Ts:.2f}s', AMB),
        (toff, tfr,  f'Tr={Tr:.2f}s', RED),
    ]:
        ax.annotate('', xy=(tb, y_ann), xytext=(ta, y_ann),
                    arrowprops=dict(arrowstyle='<->', color=clr, lw=1.2))
        ax.text((ta + tb) / 2, y_ann * 0.96, lbl,
                ha='center', va='top', color=clr,
                fontsize=8, fontfamily='monospace')

    ax.set_xlabel('Tiempo (s)', color=DIM, fontsize=10, fontfamily='monospace')
    ax.set_ylabel('Area pupilar (px2)', color=DIM, fontsize=10, fontfamily='monospace')
    ax.tick_params(colors=DIM, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color('#1e2d4a')
    ax.grid(color='#1e2d4a', linewidth=0.4, alpha=0.6)

    patches = [
        mpatches.Patch(color=CYAN, alpha=0.5, label='Basal'),
        mpatches.Patch(color=GRN,  alpha=0.5, label=f'Tc={Tc:.2f}s'),
        mpatches.Patch(color=AMB,  alpha=0.5, label=f'Ts={Ts:.2f}s'),
        mpatches.Patch(color=RED,  alpha=0.5, label=f'Tr={Tr:.2f}s'),
    ]
    ax.legend(handles=patches, fontsize=8, facecolor='#0f1828',
              labelcolor='#7a90b5', loc='upper right',
              framealpha=0.8, edgecolor='#1e2d4a')

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=110,
                bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
