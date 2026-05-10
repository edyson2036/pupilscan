/*
  PupilScan — Firmware Arduino Nano
  ──────────────────────────────────
  Espera comandos por puerto serie (9600 bps):
    "ON\n"  → enciende LED (pin 13)
    "OFF\n" → apaga LED (pin 13)

  Conexión:
    Pin 13 → Resistencia 220Ω → LED (+) → GND

  El pin 13 ya tiene un LED integrado en el Arduino Nano,
  útil para pruebas sin circuito externo.
*/

const int LED_PIN = 13;
String inputBuffer = "";

void setup() {
  Serial.begin(9600);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);   // LED apagado al inicio
  Serial.println("PupilScan Arduino OK");
}

void loop() {
  // Leer caracteres disponibles en el buffer serial
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      // Procesar comando completo
      inputBuffer.trim();
      if (inputBuffer == "ON") {
        digitalWrite(LED_PIN, HIGH);
        Serial.println("LED:ON");
      } else if (inputBuffer == "OFF") {
        digitalWrite(LED_PIN, LOW);
        Serial.println("LED:OFF");
      } else {
        Serial.println("CMD_UNKNOWN:" + inputBuffer);
      }
      inputBuffer = "";   // limpiar buffer
    } else {
      inputBuffer += c;
    }
  }
}
