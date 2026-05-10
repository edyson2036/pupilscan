"""
arduino_local.py
────────────────
Script pequeño que corre en la laptop del operador.
Expone un servidor HTTP en localhost:5001 para controlar
el LED del Arduino Nano desde el navegador.

Instalación:
    pip install flask flask-cors pyserial

Uso:
    python arduino_local.py

El navegador llama a:
    POST http://localhost:5001/led  { "state": "ON" }
    POST http://localhost:5001/led  { "state": "OFF" }
    GET  http://localhost:5001/protocol  → ejecuta protocolo completo (2s-4s-2s)
"""

import sys, time, threading, serial, serial.tools.list_ports
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # permite llamadas desde el navegador

# ── Configuración ──────────────────────────────────────────────────────────
BAUD     = 9600
TIMEOUT  = 2        # segundos timeout serial
T_BASAL  = 2.0      # segundos LED apagado antes del estímulo
T_STIM   = 4.0      # segundos LED encendido
T_RECOV  = 2.0      # segundos LED apagado después del estímulo

# ── Conexión serial ────────────────────────────────────────────────────────
arduino = None

def find_arduino():
    """Detecta automáticamente el puerto del Arduino."""
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = (p.description or '').lower()
        manu = (p.manufacturer or '').lower()
        if any(k in desc or k in manu for k in ['arduino','ch340','cp210','ftdi','uno','nano']):
            return p.device
    # Si no detecta, devuelve el primero disponible
    return ports[0].device if ports else None

def connect_arduino():
    global arduino
    port = find_arduino()
    if not port:
        print("⚠ No se encontró Arduino. El LED no funcionará.")
        return False
    try:
        arduino = serial.Serial(port, BAUD, timeout=TIMEOUT)
        time.sleep(2)  # esperar que Arduino reinicie
        print(f"✓ Arduino conectado en {port}")
        return True
    except Exception as e:
        print(f"✗ Error conectando Arduino en {port}: {e}")
        return False

def send_cmd(cmd: str):
    """Envía comando al Arduino. 'ON' enciende LED, 'OFF' apaga."""
    global arduino
    if arduino and arduino.is_open:
        try:
            arduino.write((cmd + '\n').encode())
            arduino.flush()
            print(f"  → LED {cmd}")
        except Exception as e:
            print(f"  Error serial: {e}")
            # Reintentar conexión
            connect_arduino()
    else:
        print(f"  Arduino desconectado, no se pudo enviar: {cmd}")

# ── Rutas HTTP ─────────────────────────────────────────────────────────────
@app.route('/status')
def status():
    return jsonify({
        'arduino': arduino is not None and arduino.is_open,
        'port': arduino.port if arduino else None
    })

@app.route('/led', methods=['POST'])
def led():
    data = request.get_json(force=True)
    state = data.get('state', 'OFF').upper()
    if state not in ('ON', 'OFF'):
        return jsonify({'error': 'state debe ser ON u OFF'}), 400
    send_cmd(state)
    return jsonify({'ok': True, 'state': state})

@app.route('/protocol')
def protocol():
    """
    Ejecuta el protocolo completo en un hilo separado:
      2s LED OFF (basal) → 4s LED ON (estímulo) → 2s LED OFF (recuperación)
    Devuelve inmediatamente con el tiempo de inicio.
    """
    def run():
        print("Protocolo: iniciando basal...")
        send_cmd('OFF')
        time.sleep(T_BASAL)
        print("Protocolo: LED ON (estímulo)")
        send_cmd('ON')
        time.sleep(T_STIM)
        print("Protocolo: LED OFF (recuperación)")
        send_cmd('OFF')
        time.sleep(T_RECOV)
        print("Protocolo: completo")

    t = threading.Thread(target=run, daemon=True)
    t.start()

    return jsonify({
        'ok': True,
        't_basal': T_BASAL,
        't_stim':  T_STIM,
        't_recov': T_RECOV,
        'total':   T_BASAL + T_STIM + T_RECOV
    })

@app.route('/ports')
def list_ports():
    ports = [{'device': p.device, 'desc': p.description}
             for p in serial.tools.list_ports.comports()]
    return jsonify({'ports': ports})

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print("  PupilScan — Control Arduino Local")
    print("=" * 50)
    connect_arduino()
    print(f"\nServidor en http://localhost:5001")
    print("Deja esta ventana abierta mientras usas la plataforma.\n")
    app.run(host='127.0.0.1', port=5001, debug=False)
