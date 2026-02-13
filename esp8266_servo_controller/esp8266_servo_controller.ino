/*
 * ESP8266 Servo Controller for Face-Locked Vision System
 * 
 * Subscribes to MQTT topic: vision/<team_id>/movement
 * Controls servo motor based on face movement commands:
 *   - MOVE_LEFT: rotate servo left
 *   - MOVE_RIGHT: rotate servo right
 *   - CENTERED: return servo to center position
 *   - NO_FACE: hold current position or return to center
 * 
 * Requirements:
 *   - ESP8266 board (NodeMCU, Wemos D1, etc.)
 *   - Servo motor (SG90 or similar)
 *   - WiFi network credentials
 *   - MQTT broker (VPS or local)
 * 
 * Libraries:
 *   - ESP8266WiFi
 *   - PubSubClient
 *   - Servo
 */

#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Servo.h>
#include <ArduinoJson.h>

// ========== CONFIGURATION ==========
// WiFi credentials
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// MQTT broker configuration
const char* mqtt_broker = "157.173.101.159";  // VPS IP address
const int mqtt_port = 1883;

// Team identifier (MUST be unique - change this!)
const char* team_id = "creation_squad";  // Team ID: creation_squad

// MQTT topic (automatically constructed)
char mqtt_topic_movement[64];

// Servo configuration
const int SERVO_PIN = D1;  // GPIO5 on NodeMCU (adjust for your board)
const int SERVO_CENTER = 90;  // Center position (degrees)
const int SERVO_LEFT_MAX = 0;   // Maximum left rotation
const int SERVO_RIGHT_MAX = 180; // Maximum right rotation
const int SERVO_STEP = 5;  // Degrees per movement command

// Movement thresholds (to avoid jitter)
const int CENTER_THRESHOLD = 5;  // degrees from center to consider "centered"

// ========== GLOBAL OBJECTS ==========
WiFiClient espClient;
PubSubClient client(espClient);
Servo servo;

// ========== STATE ==========
int current_servo_pos = SERVO_CENTER;
unsigned long last_movement_time = 0;
const unsigned long MOVEMENT_TIMEOUT_MS = 2000;  // 2 seconds without movement -> return to center

// ========== SETUP ==========
void setup() {
  Serial.begin(115200);
  delay(100);
  
  Serial.println("\n=== ESP8266 Servo Controller ===");
  Serial.print("Team ID: ");
  Serial.println(team_id);
  
  // Construct MQTT topic
  snprintf(mqtt_topic_movement, sizeof(mqtt_topic_movement), "vision/%s/movement", team_id);
  Serial.print("MQTT Topic: ");
  Serial.println(mqtt_topic_movement);
  
  // Initialize servo
  servo.attach(SERVO_PIN);
  servo.write(SERVO_CENTER);
  current_servo_pos = SERVO_CENTER;
  Serial.println("Servo initialized at center position");
  
  // Connect to WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("WiFi connected! IP: ");
  Serial.println(WiFi.localIP());
  
  // Setup MQTT
  client.setServer(mqtt_broker, mqtt_port);
  client.setCallback(mqtt_callback);
  
  // Connect to MQTT broker
  reconnect_mqtt();
  
  Serial.println("Setup complete. Waiting for movement commands...");
}

// ========== MAIN LOOP ==========
void loop() {
  // Maintain MQTT connection
  if (!client.connected()) {
    reconnect_mqtt();
  }
  client.loop();
  
  // Auto-return to center if no movement for timeout period
  unsigned long now = millis();
  if (now - last_movement_time > MOVEMENT_TIMEOUT_MS) {
    if (abs(current_servo_pos - SERVO_CENTER) > CENTER_THRESHOLD) {
      move_servo_toward_center();
    }
  }
  
  delay(50);  // Small delay to prevent watchdog issues
}

// ========== MQTT CALLBACK ==========
void mqtt_callback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Message received [");
  Serial.print(topic);
  Serial.print("]: ");
  
  // Parse JSON payload
  StaticJsonDocument<256> doc;
  DeserializationError error = deserializeJson(doc, payload, length);
  
  if (error) {
    Serial.print("JSON parse error: ");
    Serial.println(error.c_str());
    return;
  }
  
  const char* status = doc["status"];
  float confidence = doc["confidence"] | 0.0;
  unsigned long timestamp = doc["timestamp"] | 0;
  
  Serial.print("Status: ");
  Serial.print(status);
  Serial.print(", Confidence: ");
  Serial.println(confidence);
  
  // Update last movement time
  last_movement_time = millis();
  
  // Execute movement command
  if (strcmp(status, "MOVE_LEFT") == 0) {
    move_servo_left();
  } else if (strcmp(status, "MOVE_RIGHT") == 0) {
    move_servo_right();
  } else if (strcmp(status, "CENTERED") == 0) {
    move_servo_toward_center();
  } else if (strcmp(status, "NO_FACE") == 0) {
    // Hold position or slowly return to center
    move_servo_toward_center();
  }
}

// ========== SERVO CONTROL FUNCTIONS ==========
void move_servo_left() {
  int new_pos = current_servo_pos - SERVO_STEP;
  if (new_pos < SERVO_LEFT_MAX) {
    new_pos = SERVO_LEFT_MAX;
  }
  current_servo_pos = new_pos;
  servo.write(current_servo_pos);
  Serial.print("Servo LEFT -> ");
  Serial.println(current_servo_pos);
}

void move_servo_right() {
  int new_pos = current_servo_pos + SERVO_STEP;
  if (new_pos > SERVO_RIGHT_MAX) {
    new_pos = SERVO_RIGHT_MAX;
  }
  current_servo_pos = new_pos;
  servo.write(current_servo_pos);
  Serial.print("Servo RIGHT -> ");
  Serial.println(current_servo_pos);
}

void move_servo_toward_center() {
  if (current_servo_pos < SERVO_CENTER) {
    current_servo_pos += SERVO_STEP;
    if (current_servo_pos > SERVO_CENTER) {
      current_servo_pos = SERVO_CENTER;
    }
  } else if (current_servo_pos > SERVO_CENTER) {
    current_servo_pos -= SERVO_STEP;
    if (current_servo_pos < SERVO_CENTER) {
      current_servo_pos = SERVO_CENTER;
    }
  }
  servo.write(current_servo_pos);
  Serial.print("Servo CENTER -> ");
  Serial.println(current_servo_pos);
}

// ========== MQTT RECONNECTION ==========
void reconnect_mqtt() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    String client_id = "ESP8266_Servo_";
    client_id += String(random(0xffff), HEX);
    
    if (client.connect(client_id.c_str())) {
      Serial.println("connected!");
      Serial.print("Subscribing to: ");
      Serial.println(mqtt_topic_movement);
      client.subscribe(mqtt_topic_movement);
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" retrying in 5 seconds");
      delay(5000);
    }
  }
}
