# PupilScan — Sistema de Evaluación del Reflejo Pupilar

## Estructura
```
pupilscan/
├── app.py                    ← Backend Flask + YOLOv8 (va a Render)
├── requirements.txt          ← Dependencias Python para Render  
├── render.yaml               ← Configuración de deploy en Render
├── arduino_local.py          ← Script local para controlar el LED (corre en la laptop del operador)
├── arduino_firmware/
│   └── arduino_firmware.ino  ← Firmware para el Arduino Nano
├── static/
│   └── index.html            ← Plataforma web completa
└── best.pt                   ← Tu modelo YOLOv8n-seg (agregar manualmente)
```

## Uso rápido

### 1. Subir a Render (procesamiento en la nube)
1. Agrega `best.pt` a esta carpeta
2. Sube todo a GitHub
3. Conecta el repo en render.com → Deploy

### 2. Correr control Arduino en la laptop del operador
```bash
pip install flask flask-cors pyserial
python arduino_local.py
```

### 3. Abrir la plataforma
- Desde cualquier PC: https://TU-APP.onrender.com
- El Arduino se controla localmente desde la laptop del operador

## Protocolo
- t=0s: inicio grabación, LED apagado (basal)
- t=2s: LED encendido (estímulo)
- t=6s: LED apagado
- t=8s: fin grabación → procesamiento automático

## Parámetros extraídos
- **Tc**: tiempo de contracción pupilar (s)
- **Ts**: fase estática (s)  
- **Tr**: tiempo de recuperación (s)
- **P(t)**: curva de área pupilar en función del tiempo
