# Distributed Vision-Control System - Quick Setup Guide

**Team ID:** `creation_squad`  
**MQTT Broker:** `157.173.101.159:1883`  
**WebSocket Server:** `157.173.101.159:9002`

---

## Quick Start

### 1. PC Vision Node (Face Lock with MQTT)

```bash
python face_lock.py --lock-name "Raphael" --team-id creation_squad --mqtt-broker 157.173.101.159
```

**Note:** Default values are already set, so you can also just run:
```bash
python face_lock.py --lock-name "Raphael"
```

The system will automatically:
- Connect to MQTT broker at `157.173.101.159:1883`
- Use team ID `creation_squad`
- Publish movement states to `vision/creation_squad/movement`

### 2. ESP8266 Servo Controller

1. Open `esp8266_servo_controller/esp8266_servo_controller.ino` in Arduino IDE
2. Configure:
   - WiFi SSID and password
   - MQTT broker: `157.173.101.159` (already set)
   - Team ID: `creation_squad` (already set)
   - Servo pin: D1 (GPIO5) - adjust if needed
3. Install required libraries:
   - ESP8266WiFi
   - PubSubClient
   - Servo
   - ArduinoJson
4. Upload to ESP8266
5. Open Serial Monitor (115200 baud) to see connection status

### 3. Backend Service (MQTT → WebSocket Bridge)

```bash
cd backend
pip install -r requirements.txt
python mqtt_websocket_bridge.py --team-id creation_squad --mqtt-broker 157.173.101.159 --ws-port 9002
```

This will:
- Connect to MQTT broker at `157.173.101.159:1883`
- Subscribe to `vision/creation_squad/movement`
- Relay messages to WebSocket clients on port 9002

### 4. Web Dashboard

1. Open `dashboard/index.html` in a web browser
2. The WebSocket URL is already configured: `ws://157.173.101.159:9002`
3. You should see real-time movement updates when the face lock system is running

---

## Testing Checklist

- [ ] PC Vision Node: Face detected and locked, MQTT messages publishing
- [ ] ESP8266: Connected to WiFi, subscribed to MQTT, servo responding
- [ ] Backend Service: Connected to MQTT, WebSocket server running
- [ ] Dashboard: Connected to WebSocket, showing real-time updates

---

## Troubleshooting

**MQTT Connection Issues:**
- Verify VPS IP: `157.173.101.159`
- Check firewall allows port 1883 (MQTT)
- Ensure MQTT broker is running on the VPS

**ESP8266 Not Connecting:**
- Check WiFi credentials
- Verify team_id matches: `creation_squad`
- Check Serial Monitor for error messages

**Dashboard Not Updating:**
- Verify backend service is running
- Check browser console for WebSocket errors
- Ensure firewall allows port 9002 (WebSocket)

---

## VPS Access

**SSH Credentials:**
- Server: `157.173.101.159`
- Username: `user377`
- Password: `y4@N7!qW` (change on first login)

**Connect:**
```bash
ssh user377@157.173.101.159
```

**Note:** You may need to install and configure an MQTT broker (e.g., mosquitto) on the VPS if it's not already set up.
