import carla
import cv2
import numpy as np
import pygame
import random
import time
import threading
import math
import paho.mqtt.client as mqtt
import json
import queue
import atexit
import uuid
import os
import signal
import sys
from enum import Enum
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List
from utils import smooth_increase, max_yaw_allowed, pixel_to_angle, get_distance_to_pedestrian_centroid

@dataclass
class Pedestrian:
    x: int
    y: int
    distance: float
    time_to_collision: float
    yaw: float = 0.0
    pitch: float = 0.0
    confidence: float = 0.0

class Mode(Enum):
    KEYBOARD = 1
    STEERING_WHEEL = 2

MODE = Mode.STEERING_WHEEL
NUM_WALKERS = 75

BROKER = "localhost"
PORT = 1883
TOPIC_SEND = "vehicle"
TOPIC_REC = "action"

CAMERA_WIDTH = 1080
CAMERA_HEIGHT = 720
VIEW_FOV = 80

METRICS_PATH = "metrics_log.jsonl" 
RUN_ID = str(uuid.uuid4())
RUN_START_TS = time.time()

current_action = "normal"
level_brk = 0.05
level_intensity = 1.0
velocity = 0.0

d_min_current = float('inf')
d_min_records = []
is_braking_active = False
d_min_action = None
is_mild_brake_active = False
mild_brake_start_ts = 0.0
mild_brake_start_loc = None
speed_before_mild_brake = 0.0

TH_TTC_S = 5000   # Safe
TH_TTC_R = 2500   # Risky
TH_TTC_C = 1500   # Critical

# --- Reaction time tracking ---
ttc_safe_start = None
ttc_risky_start = None
ttc_critical_start = None
reaction_times = {
    "mild_brake": [],
    "brake": [],
    "emergency_brake": []
}
ttc_lock = threading.Lock()

brake_start_loc = None
speed_before_brake = 0.0

mqtt_client = mqtt.Client()
mqtt_queue = queue.Queue(maxsize=50)
mqtt_lock = threading.Lock()

if not os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write("")  # crea file vuoto se non esiste
try:
    mqtt_client.connect(BROKER, PORT, 60)
    mqtt_client.loop_start()
    print(f"[MQTT] Connected to {BROKER}:{PORT}")
except Exception as e:
    print("[MQTT] Connection failed:", e)

def on_mqtt_message(client, userdata, msg):
    """Callback chiamata dal thread interno MQTT.
    Non elabora nulla: mette solo il messaggio in coda.
    """
    try:
        mqtt_queue.put_nowait(msg.payload)
    except queue.Full:
        print("[MQTT] Warning: queue full, message dropped")

def mqtt_processor():
    global current_action, level_brk, level_intensity, velocity

    while True:
        try:
            payload_raw = mqtt_queue.get()
            payload = json.loads(payload_raw.decode())

            action = payload.get("action", "").lower()
            lvl = payload.get("level", None)

            if lvl is not None:
                try:
                    lvl = float(lvl)
                    lvl = max(0.0, min(lvl, 1.0))
                except ValueError:
                    lvl = 0.0
            else:
                lvl = 0.0

            if action in ["emergency_brake", "warning", "mild_brake", "brake", "normal"]:
                with mqtt_lock:
                    current_action = action
                    if current_action == "normal":
                        level_brk = 0.05
                        level_intensity = 1

            with mqtt_lock:
                if current_action == "brake":
                        level_intensity += 1
                        level_brk = smooth_increase(level_brk, level_intensity, 0.007, velocity)
            
        except Exception as e:
            print("[MQTT] Error in processor:", e)

mqtt_client.on_message = on_mqtt_message
mqtt_client.subscribe(TOPIC_REC)
processor_thread = threading.Thread(target=mqtt_processor, daemon=True)
processor_thread.start()
model = YOLO("yolov8n.pt")
cv2.namedWindow('RGB image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Depth image', cv2.WINDOW_NORMAL)

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
spectator = world.get_spectator()

PEDESTRIAN = "DET"  # DET || "RANDOM"  

weather = carla.WeatherParameters(
    cloudiness=0.0
)

# weather = carla.WeatherParameters(
#     cloudiness=90.0,         # very cloudy sky
#     precipitation=0.0,       # rain
#     precipitation_deposits=0.0, # puddles
#     wind_intensity=0.0,      # wind
#     sun_altitude_angle=-20.0, # below horizon = night
#     fog_density=0.0,         # fog density (0–100)
# )
# weather = carla.WeatherParameters(
#     cloudiness=90.0,         # very cloudy sky
#     precipitation=80.0,      # heavy rain
#     precipitation_deposits=80.0, # puddles
#     wind_intensity=60.0,     # medium-strong wind
#     sun_altitude_angle=-20.0, # below horizon = night
#     fog_density=70.0,        # fog density (0–100)
#     fog_distance=20.0,       # max visibility in meters
#     fog_falloff=1.5          # how much fog intensifies with distance
# )

# world.set_weather(weather)

metrics = {
    "run_id": RUN_ID,
    "started_at": RUN_START_TS,
    "ended_at": None,
    "scenario": {
        "num_walkers": NUM_WALKERS,
        "pedestrian_mode": PEDESTRIAN,
        "weather": {
            "cloudiness": weather.cloudiness,
            "precipitation": weather.precipitation,
            "precipitation_deposits": weather.precipitation_deposits,
            "wind_intensity": weather.wind_intensity,
            "sun_altitude_angle": weather.sun_altitude_angle,
            "fog_density": weather.fog_density,
            "fog_distance": weather.fog_distance,
            "fog_falloff": weather.fog_falloff
        }
    },
    "collisions": {
        "count": 0,
        "binary": 0,          
        "events": []         
    },
    "d_min_records": [],
    "mild_brake": [],
    "reaction_times": {
        "mild_brake": [],
        "brake": [],
        "emergency_brake": []
    },
}

if MODE == Mode.STEERING_WHEEL:
    import manual_control_steeringwheel_ubuntu as mc
    from importlib import reload
    reload(mc)
elif MODE == Mode.KEYBOARD:
    import manual_control as mc
    from importlib import reload
    reload(mc)
else:
    raise ValueError("Invalid mode selected. Choose either MODE.KEYBOARD or MODE.STEERING_WHEEL.")

input_rgb_image = None
input_rgb_image_lock = threading.Lock()

input_depth_image = None
input_depth_image_lock = threading.Lock()

processed_output = None
processed_output_lock = threading.Lock()

def remove_all(world: carla.World):
    for a in world.get_actors().filter('vehicle.*'):
        a.destroy()
    for a in world.get_actors().filter('sensor.*'):
        a.destroy()
    for a in world.get_actors().filter('walker.pedestrian.*'):
        a.destroy()
    for a in world.get_actors().filter('controller.ai.walker'):
        a.destroy()

def setup_camera(car: carla.Vehicle):
    camera_transform = carla.Transform(carla.Location(x=1.2, y=0, z=1.4), carla.Rotation(pitch=-5.0))
    blueprint_library = world.get_blueprint_library()

    rgb_bp = blueprint_library.find('sensor.camera.rgb')
    rgb_bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
    rgb_bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
    rgb_bp.set_attribute('fov', str(VIEW_FOV))
    rgb_camera = world.spawn_actor(rgb_bp, camera_transform, attach_to=car)

    depth_bp = blueprint_library.find('sensor.camera.depth')
    depth_bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
    depth_bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
    depth_bp.set_attribute('fov', str(VIEW_FOV))
    depth_camera = world.spawn_actor(depth_bp, camera_transform, attach_to=car)

    fov = float(VIEW_FOV)
    calibration = np.identity(3)
    calibration[0, 2] = CAMERA_WIDTH / 2.0
    calibration[1, 2] = CAMERA_HEIGHT / 2.0
    focal = CAMERA_WIDTH / (2.0 * np.tan(fov * np.pi / 360.0))
    calibration[0, 0] = focal
    calibration[1, 1] = focal
    rgb_camera.calibration = calibration
    depth_camera.calibration = calibration

    return rgb_camera, depth_camera

def spawn_walker(world: carla.World):
    blueprint_library = world.get_blueprint_library()

    allowed_ids = ["walker.pedestrian.0042"]
    walker_blueprints = [bp for bp in blueprint_library.filter('walker.pedestrian.*') if bp.id in allowed_ids]

    if not walker_blueprints:
        return None, None

    walker_bp = random.choice(walker_blueprints)
    if walker_bp.has_attribute("is_invincible"):
        walker_bp.set_attribute("is_invincible", "false")

    if walker_bp.has_attribute("speed"):
        speed = random.uniform(0.8, 2.0)
        walker_bp.set_attribute('speed', str(speed))
    else:
        speed = 1.0

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        return None, None

    spawn_point = random.choice(spawn_points)
    walker = world.try_spawn_actor(walker_bp, spawn_point)
    if walker is None:
        return None, None

    controller_bp = blueprint_library.find('controller.ai.walker')
    controller = world.try_spawn_actor(controller_bp, carla.Transform(), walker)
    if controller:
        controller.start()

        def random_walk():
            while controller.is_alive and walker.is_alive:
                destination = world.get_random_location_from_navigation()
                if destination:
                    controller.go_to_location(destination)
                controller.set_max_speed(speed)
                time.sleep(random.uniform(5, 15))

        threading.Thread(target=random_walk, daemon=True).start()

    return walker, controller

def spawn_walker_det(world: carla.World, existing_positions=None, min_spawn_dist=8.0):
    if existing_positions is None:
        existing_positions = []

    blueprint_library = world.get_blueprint_library()
    allowed_ids = ["walker.pedestrian.0042"]
    walker_blueprints = [
        bp for bp in blueprint_library.filter('walker.pedestrian.*') if bp.id in allowed_ids
    ]
    if not walker_blueprints:
        return None, None, existing_positions

    walker_bp = random.choice(walker_blueprints)

    if walker_bp.has_attribute("is_invincible"):
        walker_bp.set_attribute("is_invincible", "false")

    if walker_bp.has_attribute("speed"):
        speed = random.uniform(1.0, 1.6)
        walker_bp.set_attribute('speed', str(speed))
    else:
        speed = 1.2

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        return None, None, existing_positions

    random.shuffle(spawn_points)
    start_point = None
    for sp in spawn_points:
        if all(sp.location.distance(p) > min_spawn_dist for p in existing_positions):
            start_point = sp
            existing_positions.append(sp.location)
            break

    if start_point is None:
        return None, None, existing_positions

    def distance(a, b):
        return a.location.distance(b.location)

    far_points = sorted(spawn_points, key=lambda p: distance(p, start_point), reverse=True)
    farthest_point = None
    for fp in far_points:
        if fp.location.distance(start_point.location) > 20.0:  # at least 20 meters away
            farthest_point = fp
            break

    if farthest_point is None:
        farthest_point = far_points[0]

    walker = world.try_spawn_actor(walker_bp, start_point)
    if walker is None:
        return None, None, existing_positions

    controller_bp = blueprint_library.find('controller.ai.walker')
    controller = world.try_spawn_actor(controller_bp, carla.Transform(), walker)

    if controller:
        controller.start()
        controller.set_max_speed(speed)

        def walk_to_farthest():
            destination = farthest_point.location
            offset = carla.Location(
                x=random.uniform(-2.0, 2.0),
                y=random.uniform(-2.0, 2.0),
                z=0.0
            )
            destination += offset

            controller.go_to_location(destination)
            controller.set_max_speed(speed)

            while controller.is_alive and walker.is_alive:
                dist = walker.get_location().distance(destination)
                if dist < 1.0:
                    controller.stop()
                    break
                time.sleep(1.0)

        threading.Thread(target=walk_to_farthest, daemon=True).start()
    return walker, controller, existing_positions

def detect_pedestrians(image):
    device = 'cuda' 
    imgsz = 640
    results = model.predict(image, device=device, imgsz=imgsz, verbose=False)[0]

    detections = []
    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        for i in range(len(boxes)):
            if int(classes[i]) == 0 and confs[i] > 0.4:
                x1, y1, x2, y2 = boxes[i]
                bbox = (int(x1), int(y1), int(x2), int(y2))
                centroid = ((int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2)
                detections.append((confs[i], bbox, centroid))
    return detections

def send_mqtt_async(payload: dict):
    def _send():
        try:
            mqtt_client.publish(TOPIC_SEND, json.dumps(payload))
        except Exception as e:
            print("[MQTT] Publish failed:", e)
    threading.Thread(target=_send, daemon=True).start()

def get_current_action():
    with mqtt_lock:
        return current_action

def process_image():
    global input_rgb_image, input_depth_image, processed_output, velocity
    global d_min_current, d_min_records, is_braking_active, d_min_action
    global brake_start_loc, speed_before_brake
    global is_mild_brake_active, mild_brake_start_ts, mild_brake_start_loc, speed_before_mild_brake
    global ttc_safe_start, ttc_risky_start, ttc_critical_start
    global metrics
    
    last_inference_time = 0.0
    target_dt = 0.105  # 10Hz
    crossing = 0

    while True:
        with input_rgb_image_lock:
            rgb_image = input_rgb_image
        with input_depth_image_lock:
            depth_image = input_depth_image

        if rgb_image is None or depth_image is None:
            continue

        now = time.time()
        if now - last_inference_time < target_dt:
            time.sleep(0.1)
            continue
        last_inference_time = now

        rgb_array = np.frombuffer(rgb_image.raw_data, dtype=np.uint8)
        rgb_array = rgb_array.reshape((rgb_image.height, rgb_image.width, 4))[:, :, :3]

        depth_array = np.frombuffer(depth_image.raw_data, dtype=np.uint8)
        depth_array = depth_array.reshape((depth_image.height, depth_image.width, 4))

        vehicle_speed = vehicle.get_velocity()
        velocity = math.sqrt(vehicle_speed.x**2 + vehicle_speed.y**2 + vehicle_speed.z**2)

        detections = detect_pedestrians(rgb_array)
        detected_pedestrians: List[Pedestrian] = []

        for conf, _, centroid in detections:
            distance = get_distance_to_pedestrian_centroid(centroid, depth_array)
            yaw, pitch = pixel_to_angle(centroid[0], centroid[1], rgb_camera.calibration)

            time_to_collision = (distance / velocity * 1000.0) if velocity > 0.01 else float('inf')

            detected_pedestrians.append(Pedestrian(
                x=centroid[0],
                y=centroid[1],
                distance=distance,
                time_to_collision=time_to_collision,
                yaw=yaw,
                pitch=pitch,
                confidence=float(conf)
            ))

        local_action = get_current_action()
        closest_ped = min(detected_pedestrians, key=lambda p: p.distance) if detected_pedestrians else None
        conf = closest_ped.confidence if closest_ped else 0.0

        with ttc_lock:
            # === Log reaction times when state changes ===
            if local_action == "mild_brake" and ttc_safe_start is not None:
                reaction_time = time.time() - ttc_safe_start
                reaction_times["mild_brake"].append(reaction_time)
                metrics["reaction_times"]["mild_brake"].append(reaction_time)
                ttc_safe_start = None  # reset

            elif local_action == "brake" and ttc_risky_start is not None:
                reaction_time = time.time() - ttc_risky_start
                reaction_times["brake"].append(reaction_time)
                metrics["reaction_times"]["brake"].append(reaction_time)
                ttc_risky_start = None  # reset

            elif local_action == "emergency_brake" and ttc_critical_start is not None:
                reaction_time = time.time() - ttc_critical_start
                reaction_times["emergency_brake"].append(reaction_time)
                metrics["reaction_times"]["emergency_brake"].append(reaction_time)
                ttc_critical_start = None  # reset

        if closest_ped:
            yaw_rad = closest_ped.yaw
            ttc_camera = closest_ped.time_to_collision

            steer_norm = vehicle.get_control().steer  # [-1,+1]
            steer_angle_deg = steer_norm * 35.0       # +-35° max steering angle

            yaw_deg = math.degrees(yaw_rad) 
            relative_yaw = yaw_deg - steer_angle_deg
            threshold = max_yaw_allowed(closest_ped.distance)
            crossing = 1 if abs(relative_yaw) <= threshold else 0
        else:
            yaw, pitch, ttc_camera, crossing = None, None, None, 0
            
        payload = {
            "timestamp": now,
            "vehicle_speed": velocity,
            "confidence": float(conf),
            "camera_distance": closest_ped.distance if closest_ped else None,
            "camera_yaw_deg": math.degrees(yaw) if yaw is not None else None,
            "camera_pitch_deg": math.degrees(pitch) if pitch is not None else None,
            "ttc": ttc_camera if conf else 10000,
            "is_crossing": crossing if conf else 0
        }

        with ttc_lock:
            if ttc_camera and ttc_camera < float('inf'):
                if ttc_camera < TH_TTC_S and ttc_safe_start is None:
                    ttc_safe_start = time.time()
                if ttc_camera < TH_TTC_R and ttc_risky_start is None:
                    ttc_risky_start = time.time()
                if ttc_camera < TH_TTC_C and ttc_critical_start is None:
                    ttc_critical_start = time.time()

        send_mqtt_async(payload)

        closest_distance = closest_ped.distance if closest_ped else None

        if local_action in ("brake", "emergency_brake") and closest_distance:
            if not is_braking_active:
                is_braking_active = True
                d_min_current = float('inf')
                brake_start_loc = vehicle.get_location()
                speed_before_brake = velocity

            if closest_distance < d_min_current:
                d_min_current = closest_distance
                d_min_action = local_action

        elif is_braking_active and local_action not in ("brake", "emergency_brake"):
            stop_loc = vehicle.get_location()
            stopping_distance = brake_start_loc.distance(stop_loc) if brake_start_loc and stop_loc else None

            d_min_records.append({
                "timestamp": time.time(),
                "action": d_min_action if d_min_action else "unknown",
                "d_min": d_min_current if d_min_current != float('inf') else None,
                "stopping_distance": stopping_distance if stopping_distance else None,
                "speed_before_brake": speed_before_brake if speed_before_brake else None
            })
            is_braking_active = False
            d_min_current = float('inf')
            brake_start_loc = None
            stopping_distance = None
            speed_before_brake = 0.0

        # --- Mild Brake logging (not emergency, just release of throttle) ---
        if local_action == "mild_brake":
            if not is_mild_brake_active:
                is_mild_brake_active = True
                mild_brake_start_ts = time.time()
                mild_brake_start_loc = vehicle.get_location()
                speed_before_mild_brake = velocity
        else:
            if is_mild_brake_active:
                # Mild brake has just ended
                stop_loc = vehicle.get_location()
                distance_travelled = mild_brake_start_loc.distance(stop_loc) if mild_brake_start_loc and stop_loc else None
                duration = time.time() - mild_brake_start_ts

                metrics["mild_brake"].append({
                    "timestamp": time.time(),
                    "speed_before_mild_brake": speed_before_mild_brake,
                    "duration": duration,
                    "distance_travelled": distance_travelled
                })

                is_mild_brake_active = False
                mild_brake_start_ts = 0.0
                mild_brake_start_loc = None
                speed_before_mild_brake = 0.0
        with processed_output_lock:
            processed_output = {
                "rgb_image": rgb_array,
                "depth_image": depth_array,
                "detections": detections
            }

class GameLoop(object):
    def __init__(self, args):
        self.args = args
        pygame.init()
        pygame.font.init()
        self.world = None
        self.original_settings = None
        self.fps = args.maxfps

        try:
            self.sim_world = client.get_world()
            if args.sync:
                self.original_settings = self.sim_world.get_settings()
                settings = self.sim_world.get_settings()
                if not settings.synchronous_mode:
                    settings.synchronous_mode = True
                    settings.fixed_delta_seconds = 0.05
                self.sim_world.apply_settings(settings)

                traffic_manager = client.get_trafficmanager()
                traffic_manager.set_synchronous_mode(True)

            if not self.sim_world.get_settings().synchronous_mode:
                pass

            self.display = pygame.display.set_mode(
                (args.width, args.height),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.display.fill((0,0,0))
            pygame.display.flip()

            hud = mc.HUD(args.width, args.height)
            self.world = mc.World(self.sim_world, hud, args)
            self.controller = None
            if MODE == Mode.STEERING_WHEEL:
                self.controller = mc.DualControl(self.world)
            elif MODE == Mode.KEYBOARD:
                self.controller = mc.KeyboardControl()

            if args.sync:
                self.sim_world.tick()
            else:
                self.sim_world.wait_for_tick()
        except Exception:
            mc.logging.exception('Error creating the world')

    def render(self, clock: pygame.time.Clock):
        self.world.tick(clock)
        self.world.render(self.display)
        pygame.display.flip()

    def start(self):
        self.world.player.set_autopilot(False)
        try:
            clock = pygame.time.Clock()
            global current_action

            while True:
                # Sync tick 
                if self.args.sync:
                    self.sim_world.tick()
                clock.tick_busy_loop(self.fps)
                with mqtt_lock:
                    local_action = current_action
                    local_level_brk = level_brk

                try:
                    ready = False
                    with processed_output_lock:
                        if processed_output is not None \
                        and "rgb_image" in processed_output \
                        and "depth_image" in processed_output \
                        and "detections" in processed_output:
                            rgb_arr = processed_output["rgb_image"]
                            depth_arr = processed_output["depth_image"]
                            dets = processed_output["detections"]
                            ready = True

                    if ready:
                        if rgb_arr is not None and rgb_arr.ndim == 3 and rgb_arr.shape[2] >= 3:
                            bgr_for_display = cv2.cvtColor(rgb_arr[:, :, :3], cv2.COLOR_RGB2BGR)
                        else:
                            bgr_for_display = None

                        if depth_arr is not None and depth_arr.ndim == 3 and depth_arr.shape[2] >= 4:
                            rgb = depth_arr[:, :, :3].astype(np.uint32)
                            r = rgb[:, :, 2]
                            g = rgb[:, :, 1]
                            b = rgb[:, :, 0]
                            norm = (r + g * 256 + b * 256**2) / (256**3 - 1)
                            depth_m = np.clip(norm * 1000.0, 0, 50)
                            depth_vis = (255 * (1.0 - depth_m / 50.0)).astype(np.uint8)
                            depth_for_display = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

                            h, _, _ = depth_for_display.shape
                            legend_h = h
                            legend_w = 40
                            legend = np.linspace(255, 0, legend_h).astype(np.uint8)
                            legend = cv2.applyColorMap(legend.reshape(-1, 1), cv2.COLORMAP_JET)
                            legend = cv2.resize(legend, (legend_w, legend_h))

                            for _, dist in enumerate([0, 10, 20, 30, 40, 50]):
                                y = int(legend_h - (dist / 50.0) * legend_h)
                                cv2.putText(legend, f"{dist}m", (2, max(12, y - 2)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                            depth_for_display = np.hstack((depth_for_display, legend))
                        else:
                            depth_for_display = None

                        if bgr_for_display is not None and dets:
                            for _, bbox, _ in dets:
                                cv2.rectangle(
                                    bgr_for_display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2
                                )
                            steer_angle_deg = vehicle.get_control().steer * 35.0
                            cv2.line(
                                bgr_for_display,
                                (CAMERA_WIDTH // 2, CAMERA_HEIGHT),
                                (CAMERA_WIDTH // 2 + int(math.tan(math.radians(steer_angle_deg)) * 300),
                                CAMERA_HEIGHT - 300),
                                (0, 0, 255), 2
                            )

                        if bgr_for_display is not None:
                            cv2.imshow('RGB image', bgr_for_display)
                        if depth_for_display is not None:
                            cv2.imshow('Depth image', depth_for_display)

                        cv2.waitKey(1)

                except Exception as e:
                    print(f"[DISPLAY] Error updating OpenCV windows: {e}")

                if local_action == "normal":
                    self.controller.parse_events(self.world, clock)

                current_control = vehicle.get_control()
                control = carla.VehicleControl()

                control.steer = current_control.steer

                if local_action == "emergency_brake":
                    control.throttle = 0.0
                    control.brake = 1.0
                    vehicle.apply_control(control)
                    # print("[MQTT] Frenata di emergenza attiva")

                elif local_action == "brake":
                    control.throttle = 0.0
                    control.brake = local_level_brk
                    vehicle.apply_control(control)
                    # print(f"[MQTT] Frenata lieve applicata (level={control.brake:.2f})")

                elif local_action == "mild_brake":
                    control.throttle = 0.0
                    control.brake = 0.0
                    vehicle.apply_control(control)
                    # print("[MQTT] Rilasciato acceleratore")

                elif local_action == "warning":
                    self.controller.parse_events(self.world, clock)
                    # print("[MQTT] Avviso al conducente: pedone vicino")
                
                self.render(clock)

        finally:
            if self.original_settings:
                self.sim_world.apply_settings(self.original_settings)
            if self.world is not None:
                self.world.destroy()
            pygame.quit()

def setup():
    argparser = mc.argparse.ArgumentParser(description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1", "2", "All" - default: "2")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Activate synchronous mode execution')
    argparser.add_argument(
        '--maxfps',
        default=30,
        type=int,
        help='Fps of the client (default: 30)')
    args = argparser.parse_args()

    args.res = '1280x720'
    args.width, args.height = CAMERA_WIDTH, CAMERA_HEIGHT
    args.filter = 'vehicle.mercedes.coupe_2020'
    args.sync = True

    log_level = mc.logging.DEBUG if args.debug else mc.logging.INFO
    mc.logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    mc.logging.info('listening to server %s:%s', args.host, args.port)

    return GameLoop(args)

existing_positions = []
if PEDESTRIAN == "DET":
    for _ in range(NUM_WALKERS):
        _, _, existing_positions = spawn_walker_det(world, existing_positions)
else:
    for _ in range(NUM_WALKERS):
        _, _ = spawn_walker(world)

game_loop = setup()
vehicle = world.get_actors().filter('vehicle.*')[0]
rgb_camera, depth_camera = setup_camera(vehicle)

collision_data = {
    "count": 0,
    "last_time": 0.0,
    "last_actor": None,
    "cooldown": 10.0
}

def collision_callback(event):
    global collision_data, metrics
    other_actor = event.other_actor
    actor_id = other_actor.id
    actor_type = other_actor.type_id

    impulse = event.normal_impulse
    magnitude = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
    now = time.time()

    if (actor_id == collision_data["last_actor"]) and (now - collision_data["last_time"] < collision_data["cooldown"]):
        return

    collision_data["count"] += 1
    collision_data["last_time"] = now
    collision_data["last_actor"] = actor_id

    metrics["collisions"]["count"] = collision_data["count"]
    metrics["collisions"]["binary"] = 1
    metrics["collisions"]["events"].append({
        "t": now,
        "other_actor": actor_type,
        "intensity": magnitude
    })

    print(f"[COLLISION] #{collision_data['count']} with {actor_type}, intensity={magnitude:.2f}")

blueprint_library = world.get_blueprint_library()
collision_bp = blueprint_library.find('sensor.other.collision')
collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
collision_sensor.listen(lambda event: collision_callback(event))

def rgb_camera_callback(image):
    try:
        global input_rgb_image
        with input_rgb_image_lock:
            input_rgb_image = image
    except Exception as e:
        pass

def depth_camera_callback(image):
    try:
        global input_depth_image
        with input_depth_image_lock:
            input_depth_image = image
    except Exception as e:
        pass

rgb_camera.listen(lambda image: rgb_camera_callback(image))
depth_camera.listen(lambda image: depth_camera_callback(image))

def _append_metrics_to_file():
    try:
        metrics["ended_at"] = metrics.get("ended_at") or time.time()
        metrics["d_min_records"] = d_min_records
        metrics["reaction_times"] = reaction_times
        payload = json.dumps(metrics, ensure_ascii=False)
        with open(METRICS_PATH, "a", encoding="utf-8") as f:
            f.write(payload + "\n")
        print(f"[METRICS] Appended run metrics to {os.path.abspath(METRICS_PATH)}")
    except Exception as e:
        print("[METRICS] Failed to append metrics:", e)

def _graceful_exit_handler(signum=None, frame=None):
    if metrics.get("_flushed"):
        return
    metrics["_flushed"] = True
    _append_metrics_to_file()

    print(f"[EXIT] Received signal {signum}, shutting down...")

    try:
        cv2.destroyAllWindows()
        pygame.quit()
    except Exception:
        pass

    sys.exit(0)

atexit.register(_graceful_exit_handler)
for _sig in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(_sig, _graceful_exit_handler)
    except Exception:
        pass

threading.Thread(target=process_image, daemon=True).start()

def cleanup():
    try:
        metrics["ended_at"] = time.time()
        if not metrics.get("_flushed"):
            metrics["_flushed"] = True
            _append_metrics_to_file()
    finally:
        remove_all(world)

try:
    game_loop.start()
except KeyboardInterrupt:
    print("\n[EXIT] Keyboard interrupt, cleaning up...")
finally:
    cleanup()