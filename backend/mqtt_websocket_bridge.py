#!/usr/bin/env python3
"""
Backend API Service: MQTT → WebSocket Bridge

This service:
1. Hosts an MQTT broker (via mosquitto or connects to external broker)
2. Subscribes to vision/<team_id>/movement topics
3. Relays movement messages to connected WebSocket clients
4. Runs on port 9002 (WebSocket) and connects to MQTT broker on port 1883

Requirements:
    pip install paho-mqtt websockets asyncio-mqtt

Usage:
    python backend/mqtt_websocket_bridge.py --team-id creation_squad --mqtt-broker 157.173.101.159 --mqtt-port 1883 --ws-port 9002

Note: This script connects to an existing MQTT broker. For a full setup, you may need
to run mosquitto separately or use a cloud MQTT broker.
"""

import argparse
import asyncio
import json
import logging
import time
from typing import Set

try:
    import paho.mqtt.client as mqtt
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install paho-mqtt websockets")
    exit(1)

# ========== CONFIGURATION ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
connected_clients: Set[WebSocketServerProtocol] = set()
team_id: str = ""
mqtt_topic_movement: str = ""


# ========== MQTT CALLBACKS ==========
def on_mqtt_connect(client, userdata, flags, rc):
    """Called when MQTT client connects to broker."""
    if rc == 0:
        logger.info(f"MQTT connected to broker")
        logger.info(f"Subscribing to: {mqtt_topic_movement}")
        client.subscribe(mqtt_topic_movement, qos=1)
    else:
        logger.error(f"MQTT connection failed with code {rc}")


def on_mqtt_message(client, userdata, msg):
    """Called when MQTT message is received."""
    try:
        topic = msg.topic
        payload = msg.payload.decode('utf-8')
        logger.debug(f"MQTT message [{topic}]: {payload}")
        
        # Parse JSON payload
        data = json.loads(payload)
        
        # Broadcast to all WebSocket clients
        asyncio.create_task(broadcast_to_websockets(data))
    except Exception as e:
        logger.error(f"Error processing MQTT message: {e}")


def on_mqtt_disconnect(client, userdata, rc):
    """Called when MQTT client disconnects."""
    logger.warning("MQTT disconnected")


# ========== WEBSOCKET HANDLERS ==========
async def broadcast_to_websockets(data: dict):
    """Broadcast movement data to all connected WebSocket clients."""
    if not connected_clients:
        return
    
    message = json.dumps({
        "type": "movement",
        "data": data,
        "timestamp": time.time()
    })
    
    disconnected = set()
    for client in connected_clients:
        try:
            await client.send(message)
        except Exception as e:
            logger.warning(f"Failed to send to WebSocket client: {e}")
            disconnected.add(client)
    
    # Remove disconnected clients
    connected_clients.difference_update(disconnected)


async def websocket_handler(websocket: WebSocketServerProtocol, path: str):
    """Handle new WebSocket connection."""
    client_addr = websocket.remote_address
    logger.info(f"WebSocket client connected: {client_addr}")
    connected_clients.add(websocket)
    
    try:
        # Send welcome message
        welcome = json.dumps({
            "type": "welcome",
            "team_id": team_id,
            "timestamp": time.time()
        })
        await websocket.send(welcome)
        
        # Keep connection alive and handle incoming messages (if any)
        async for message in websocket:
            # Echo or handle client messages if needed
            logger.debug(f"Received from {client_addr}: {message}")
            
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"WebSocket client disconnected: {client_addr}")
    except Exception as e:
        logger.error(f"WebSocket error for {client_addr}: {e}")
    finally:
        connected_clients.discard(websocket)


# ========== MQTT CLIENT SETUP ==========
def setup_mqtt_client(broker: str, port: int):
    """Create and configure MQTT client."""
    client = mqtt.Client(client_id=f"bridge_{team_id}")
    client.on_connect = on_mqtt_connect
    client.on_message = on_mqtt_message
    client.on_disconnect = on_mqtt_disconnect
    
    try:
        logger.info(f"Connecting to MQTT broker: {broker}:{port}")
        client.connect(broker, port, keepalive=60)
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MQTT broker: {e}")
        raise


# ========== MAIN ==========
async def main():
    """Main async function."""
    global team_id, mqtt_topic_movement
    
    parser = argparse.ArgumentParser(description="MQTT → WebSocket Bridge")
    parser.add_argument("--team-id", type=str, required=True,
                        help="Team identifier (e.g. team01, alpha)")
    parser.add_argument("--mqtt-broker", type=str, default="localhost",
                        help="MQTT broker hostname/IP")
    parser.add_argument("--mqtt-port", type=int, default=1883,
                        help="MQTT broker port")
    parser.add_argument("--ws-port", type=int, default=9002,
                        help="WebSocket server port")
    parser.add_argument("--ws-host", type=str, default="0.0.0.0",
                        help="WebSocket server host (0.0.0.0 for all interfaces)")
    
    args = parser.parse_args()
    team_id = args.team_id
    mqtt_topic_movement = f"vision/{team_id}/movement"
    
    logger.info("=" * 60)
    logger.info("MQTT → WebSocket Bridge Service")
    logger.info(f"Team ID: {team_id}")
    logger.info(f"MQTT Topic: {mqtt_topic_movement}")
    logger.info(f"MQTT Broker: {args.mqtt_broker}:{args.mqtt_port}")
    logger.info(f"WebSocket Server: ws://{args.ws_host}:{args.ws_port}")
    logger.info("=" * 60)
    
    # Setup MQTT client
    mqtt_client = setup_mqtt_client(args.mqtt_broker, args.mqtt_port)
    mqtt_client.loop_start()
    
    # Start WebSocket server
    logger.info(f"Starting WebSocket server on ws://{args.ws_host}:{args.ws_port}")
    async with websockets.serve(websocket_handler, args.ws_host, args.ws_port):
        logger.info("WebSocket server started. Waiting for connections...")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
