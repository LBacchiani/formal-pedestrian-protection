import json
import paho.mqtt.client as mqtt
from automaton import PedestrianProtectionAutomaton  # assuming your automaton is in this module
import time

# ============================================================================
# MQTT CONFIGURATION
# ============================================================================
BROKER_ADDRESS = "localhost"  # or the IP of your Mosquitto broker
BROKER_PORT = 1883
TOPIC = "vehicle"
PUB_TOPIC = "action"

# ============================================================================
# AUTOMATON INSTANCE
# ============================================================================
automaton = PedestrianProtectionAutomaton()

# ============================================================================
# CALLBACKS
# ============================================================================

def on_connect(client, userdata, flags, rc):
    """Called when the client connects to the broker."""
    if rc == 0:
        print("Connected to broker successfully!")
        client.subscribe(TOPIC)
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    """Called when a message is received on a subscribed topic."""
    try:
        payload = msg.payload.decode('utf-8')
        data = json.loads(payload)
        
        # Extract expected fields
        ric = time.time() - data.get("timestamp") 
        confidence = data.get("confidence")      # float [0,1]
        ttc = data.get("ttc")                    # float seconds
        is_crossing = data.get("is_crossing")    # int {0,1}
        # print(f"Received data: confidence={confidence}, ttc={ttc}, is_crossing={is_crossing}")
        
        st = time.time()
        # Update automaton with new data
        automaton.update_data(confidence=confidence, ttc=ttc, is_crossing=is_crossing)
        # Perform a step
        action = automaton.step()
        aut = time.time() - st
        
        # Print or handle action
        # print(f"Automaton state: {automaton.state.value}, Action: {action.value}")
        sen = time.time()
        client.publish(PUB_TOPIC, json.dumps({"action": action.value, "send": ric, "aut": aut, "ret": sen}))

    
    except Exception as e:
        print(f"Error processing message: {e}")

# ============================================================================
# MQTT CLIENT SETUP
# ============================================================================
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# Connect to broker
client.connect(BROKER_ADDRESS, BROKER_PORT, keepalive=60)

# Start the MQTT client loop
client.loop_forever()
