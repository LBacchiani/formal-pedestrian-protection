import carla
import cv2
import numpy as np
import pygame
from ultralytics import YOLO
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
import time
import threading
import math
import torch
from typing import Tuple
import requests
import paho.mqtt.client as mqtt
import json


#########################################
################ Classes ################
#########################################

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

##########################################
################# Config #################
##########################################

MODE = Mode.KEYBOARD
CAMERA_DEBUG = True
NUM_WALKERS = 75

BROKER = "localhost"
PORT = 1883
TOPIC_SEND = "vehicle"
TOPIC_REC = "action"

CAMERA_WIDTH = 1080
CAMERA_HEIGHT = 720
VIEW_FOV = 80

# Stato attuale del veicolo in base ai messaggi MQTT
current_action = "normal"
level_brk = 0.0
mqtt_last_update = 0.0


# ---- MQTT setup ----
mqtt_client = mqtt.Client()
try:
    mqtt_client.connect(BROKER, PORT, 60)
    mqtt_client.loop_start()
    print(f"[MQTT] Connected to {BROKER}:{PORT}")
except Exception as e:
    print("[MQTT] Connection failed:", e)

def on_mqtt_message(client, userdata, msg):
    global current_action, mqtt_last_update, level_brk

    try:
        payload = json.loads(msg.payload.decode())
        action = payload.get("action", "").lower()
        # Livello di frenata (solo se presente e valido)
        lvl = payload.get("level", None)
        if lvl is not None:
            try:
                level_brk = float(lvl)
                level_brk = max(0.0, min(level_brk, 1.0)) 
            except ValueError:
                level_brk = 0.0
        else:
            level_brk = 0.0

        print(str(payload))
        print(str(action))
        if action in ["emergency_brake", "warning", "mild_brake", "normal", "_"]:
            if action != "_":
                current_action = action
                mqtt_last_update = time.time()
                print(f"[MQTT] Action received: {action}")
        else:
            print(f"[MQTT] Unknown action: {action}")
    except Exception as e:
        print("[MQTT] Error parsing message:", e)

mqtt_client.on_message = on_mqtt_message
mqtt_client.subscribe(TOPIC_REC)


model = YOLO("yolov8n.pt")

if CAMERA_DEBUG:
    cv2.namedWindow('RGB image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Depth image', cv2.WINDOW_NORMAL)

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
spectator = world.get_spectator()

PEDESTRIAN = "DET"  # DET || "RANDOM"  

### Modalità Sole ### 

### Modalità notte
# weather = carla.WeatherParameters(
#     cloudiness=90.0,         # cielo molto coperto
#     precipitation=0.0,      # pioggia
#     precipitation_deposits=0.0, # pozzanghere
#     wind_intensity=0.0,        #vento 
#     sun_altitude_angle=-20.0, # sotto l'orizzonte = notte
#     fog_density=0.0,        # densità nebbia (0–100)
# )
### Modalità Notte con nebbia
# weather = carla.WeatherParameters(
#     cloudiness=90.0,         # cielo molto coperto
#     precipitation=80.0,      # pioggia forte
#     precipitation_deposits=80.0, # pozzanghere
#     wind_intensity=60.0,     # vento medio-forte
#     sun_altitude_angle=-20.0, # sotto l'orizzonte = notte
#     fog_density=70.0,        # densità nebbia (0–100)
#     fog_distance=20.0,       # visibilità max in metri
#     fog_falloff=1.5          # quanto la nebbia si intensifica con la distanza
# )



# Applica al mondo
world = client.get_world()
# world.set_weather(weather)

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

###########################################
############ Utility functions ############
###########################################
def emergency_brake():
    global vehicle
    control = carla.VehicleControl(throttle=0.0, brake=1.0)
    vehicle.apply_control(control)


def remove_all(world: carla.World):
    """
    Removes all actors from the CARLA world, including vehicles, sensors, and pedestrians.
    Args:
        world (carla.World): The CARLA world object from which actors will be removed.
    """
    for a in world.get_actors().filter('vehicle.*'):
        a.destroy()
    for a in world.get_actors().filter('sensor.*'):
        a.destroy()
    for a in world.get_actors().filter('walker.pedestrian.*'):
        a.destroy()
    for a in world.get_actors().filter('controller.ai.walker'):
        a.destroy()


def carla_transform_to_matrix(transform: carla.Transform) -> np.ndarray:
    """
    Converte carla.Transform in matrice 4x4 (mondo <- oggetto).
    Rotazioni in ordine yaw(Z) -> pitch(Y) -> roll(X), sistema CARLA (x fwd, y right, z up).
    """
    r = transform.rotation
    t = transform.location

    cy = math.cos(math.radians(r.yaw))
    sy = math.sin(math.radians(r.yaw))
    cp = math.cos(math.radians(r.pitch))
    sp = math.sin(math.radians(r.pitch))
    cr = math.cos(math.radians(r.roll))
    sr = math.sin(math.radians(r.roll))

    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    R = np.array([
        [cp*cy, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [cp*sy, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [  -sp,            cp*sr,            cp*cr]
    ])
    M = np.eye(4)
    M[0:3, 0:3] = R
    M[0:3, 3] = np.array([t.x, t.y, t.z])
    return M


def world_to_pixel(
    loc_world: carla.Location,
    camera: carla.Sensor,
    K: np.ndarray
) -> Optional[Tuple[int, int]]:
    M_cam = carla_transform_to_matrix(camera.get_transform())
    world2cam = np.linalg.inv(M_cam)

    p_world = np.array([loc_world.x, loc_world.y, loc_world.z, 1.0])
    p_cam = world2cam @ p_world  # [X, Y, Z] con X forward, Y right, Z up

    X, Y, Z = p_cam[0], p_cam[1], p_cam[2]
    if X <= 0:
        return None  # dietro la camera

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Proiezione pinhole
    u = cx + fx * (Y / X)
    v = cy - fy * (Z / X)  # attenzione segno verticale

    if 0 <= u < CAMERA_WIDTH and 0 <= v < CAMERA_HEIGHT:
        return int(u), int(v)
    return None

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

    # usa il FOV reale impostato
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

    allowed_ids = ["walker.pedestrian.0042", "walker.pedestrian.0039", "walker.pedestrian.0037"]
    walker_blueprints = [bp for bp in blueprint_library.filter('walker.pedestrian.*') if bp.id in allowed_ids]

    if not walker_blueprints:
        return None, None

    walker_bp = random.choice(walker_blueprints)

    # assegna velocità
    if walker_bp.has_attribute("speed"):
        speed = random.uniform(0.8, 2.0)
        walker_bp.set_attribute('speed', str(speed))
    else:
        speed = 1.0

    # spawn casuale
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        return None, None

    spawn_point = random.choice(spawn_points)
    walker = world.try_spawn_actor(walker_bp, spawn_point)
    if walker is None:
        return None, None

    # aggiungi controller AI
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
    allowed_ids = ["walker.pedestrian.0042", "walker.pedestrian.0039", "walker.pedestrian.0037"]
    walker_blueprints = [
        bp for bp in blueprint_library.filter('walker.pedestrian.*') if bp.id in allowed_ids
    ]
    if not walker_blueprints:
        return None, None, existing_positions

    walker_bp = random.choice(walker_blueprints)
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

    # Trova il punto più lontano tra quelli non troppo vicini
    def distance(a, b):
        return a.location.distance(b.location)

    far_points = sorted(spawn_points, key=lambda p: distance(p, start_point), reverse=True)
    farthest_point = None
    for fp in far_points:
        if fp.location.distance(start_point.location) > 20.0:  # almeno 20m di cammino
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

            # Piccola deviazione casuale per evitare traiettorie perfettamente allineate
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

##########################################
############ Image Processing ############
##########################################

def detect_pedestrians(image):
    """
    Rileva pedoni in un'immagine usando YOLOv8.
    - Usa automaticamente GPU se disponibile
    - Se è su CPU, riduce la risoluzione (imgsz=480)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    imgsz = 640 if device == 'cuda' else 480
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

def get_distance_to_pedestrian_centroid(centroid, depth_image):
    x, y = centroid

    blue  = depth_image[y, x, 0]
    green = depth_image[y, x, 1]
    red   = depth_image[y, x, 2]

    normalized_depth = (red + green * 256 + blue * 256**2) / (256**3 - 1)

    depth_in_meters = normalized_depth * 1000.0

    return depth_in_meters

def pixel_to_angle(u: int, v: int, K: np.ndarray) -> Tuple[float, float]:
    """
    Converte un pixel immagine (u,v) in angolo orizzontale (yaw) e verticale (pitch)
    rispetto all'asse ottico della camera.

    Args:
        u (int): coordinata x del pixel
        v (int): coordinata y del pixel
        K (np.ndarray): matrice intrinseca 3x3 della camera

    Returns:
        (yaw, pitch) in radianti
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # normalizzazione pixel
    x = (u - cx) / fx
    y = (v - cy) / fy

    # direzione nel frame camera (z forward)
    vec = np.array([1.0, x, -y])  
    vec /= np.linalg.norm(vec)     # normalizza

    # angoli rispetto all'asse z (forward)
    yaw = math.atan2(vec[1], vec[0])    # deviazione orizzontale
    pitch = math.atan2(vec[2], vec[0])  # deviazione verticale

    return yaw, pitch

def send_mqtt_async(payload: dict):
    def _send():
        try:
            mqtt_client.publish(TOPIC_SEND, json.dumps(payload))
        except Exception as e:
            print("[MQTT] Publish failed:", e)
    threading.Thread(target=_send, daemon=True).start()


def max_yaw_allowed(distance):
    """
    Restituisce l'angolo massimo (in gradi) che consideriamo crossing.
    - 0 m  → ±35°
    - 25 m → ±0° (oltre nessun crossing)
    """
    if distance <= 0:
        return 35.0
    elif distance >= 25.0:
        return 0.0
    else:
        # Interpolazione lineare da 35° (0 m) → 0° (25 m)
        return 35.0 * (1 - distance / 25.0)


def process_image():
    """
    Thread di elaborazione immagini:
      - converte i frame RGB/Depth in numpy (zero-copy)
      - esegue YOLO ogni target_dt
      - calcola distanza, TTC e flag crossing
      - salva il risultato per il rendering e la logica ADAS
    """
    global input_rgb_image, input_depth_image, processed_output

    last_inference_time = 0.0
    target_dt = 0.10  # 10Hz
    crossing = 0

    print("[PROCESS] Avviato thread di elaborazione immagini...")

    while True:
        # acquisizione immagini
        with input_rgb_image_lock:
            rgb_image = input_rgb_image
        with input_depth_image_lock:
            depth_image = input_depth_image

        if rgb_image is None or depth_image is None:
            time.sleep(0.02)
            continue

        now = time.time()
        if now - last_inference_time < target_dt:
            time.sleep(0.01)
            continue
        last_inference_time = now

        # conversione zero-copy
        rgb_array = np.frombuffer(rgb_image.raw_data, dtype=np.uint8)
        rgb_array = rgb_array.reshape((rgb_image.height, rgb_image.width, 4))[:, :, :3]

        depth_array = np.frombuffer(depth_image.raw_data, dtype=np.uint8)
        depth_array = depth_array.reshape((depth_image.height, depth_image.width, 4))


        # velocità veicolo
        vehicle_speed = vehicle.get_velocity()
        vehicle_speed_mps = math.sqrt(vehicle_speed.x**2 + vehicle_speed.y**2 + vehicle_speed.z**2)

        # detection pedoni (YOLO)

        detections = detect_pedestrians(rgb_array)

        detected_pedestrians: List[Pedestrian] = []

        # elaborazione pedoni
        for conf, _, centroid in detections:
            distance = get_distance_to_pedestrian_centroid(centroid, depth_array)
            yaw, pitch = pixel_to_angle(centroid[0], centroid[1], rgb_camera.calibration)

            # ottieni angolo di sterzo del veicolo
            steer_norm = vehicle.get_control().steer  # range [-1, +1]
            steer_angle_deg = steer_norm * 35.0       # ±35° sterzo max

            # calcola deviazione yaw rispetto alla direzione del volante
            yaw_deg = math.degrees(yaw)
            relative_yaw = yaw_deg - steer_angle_deg  # ruota il cono nella direzione di sterzo

            threshold = max_yaw_allowed(distance)
            crossing = 1 if abs(relative_yaw) <= threshold else 0

            time_to_collision = distance / vehicle_speed_mps if vehicle_speed_mps > 0.01 else float('inf')

            detected_pedestrians.append(Pedestrian(
                x=centroid[0],
                y=centroid[1],
                distance=distance,
                time_to_collision=time_to_collision,
                yaw=yaw,
                pitch=pitch,
                confidence=float(conf)
            ))

        # trova pedone più vicino
        closest_ped = min(detected_pedestrians, key=lambda p: p.distance) if detected_pedestrians else None

        # confidenza del pedone più vicino
        conf = closest_ped.confidence if closest_ped else 0.0

        # angoli e TTC del più vicino
        if closest_ped:
            yaw, pitch = closest_ped.yaw, closest_ped.pitch
            ttc_camera = closest_ped.time_to_collision
            # crossing coerente col più vicino
            yaw_deg_closest = abs(math.degrees(yaw))
            crossing = 1 if yaw_deg_closest <= max_yaw_allowed(closest_ped.distance) else 0
        else:
            yaw, pitch, ttc_camera, crossing = None, None, None, 0
            
        # prepara payload
        payload = {
            "timestamp": now,
            "vehicle_speed": vehicle_speed_mps,
            "confidence": float(conf),
            "camera_distance": closest_ped.distance if closest_ped else None,
            "camera_yaw_deg": math.degrees(yaw) if yaw is not None else None,
            "camera_pitch_deg": math.degrees(pitch) if pitch is not None else None,
            "ttc": ttc_camera if conf else 10000,
            "is_crossing": crossing if conf else 0
        }

        send_mqtt_async(payload)

        # # stampa di debug
        # print("[SUMMARY]",
        #       f"speed={vehicle_speed_mps:.2f} m/s",
        #       f"confidence={payload['confidence']:.2f}",
        #       f"dist={payload['camera_distance']:.1f}m" if payload['camera_distance'] else "dist=None",
        #       f"yaw={payload['camera_yaw_deg']:.1f}°" if payload['camera_yaw_deg'] else "yaw=None",
        #       f"ttc={payload['ttc']:.2f}" if payload['ttc'] else "ttc=None",
        #       f"is_crossing={payload['is_crossing']}"
        # )

        # salva per rendering
        with processed_output_lock:
            processed_output = {
                "rgb_image": rgb_array,
                "depth_image": depth_array,
                "detections": detections
            }

##########################################
########### Gameloop and Setup ###########
##########################################

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
            global current_action, mqtt_last_update

            while True:
                # Sync tick 
                if self.args.sync:
                    self.sim_world.tick()
                clock.tick_busy_loop(self.fps)

                # Visualizzazione OpenCV
                if CAMERA_DEBUG:
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
                            # RGB: da RGB (CARLA) a BGR (OpenCV)
                            if rgb_arr is not None and rgb_arr.ndim == 3 and rgb_arr.shape[2] >= 3:
                                bgr_for_display = cv2.cvtColor(rgb_arr[:, :, :3], cv2.COLOR_RGB2BGR)
                            else:
                                bgr_for_display = None

                            # DEPTH: converti da codifica CARLA a visuale colorata
                            if depth_arr is not None and depth_arr.ndim == 3 and depth_arr.shape[2] >= 4:
                                rgb = depth_arr[:, :, :3].astype(np.uint32)
                                r = rgb[:, :, 2]
                                g = rgb[:, :, 1]
                                b = rgb[:, :, 0]
                                norm = (r + g * 256 + b * 256**2) / (256**3 - 1)
                                depth_m = np.clip(norm * 1000.0, 0, 50)
                                depth_vis = (255 * (1.0 - depth_m / 50.0)).astype(np.uint8)
                                depth_for_display = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

                                # Aggiungi legenda laterale (color bar) 
                                h, w, _ = depth_for_display.shape
                                legend_h = h
                                legend_w = 40
                                legend = np.linspace(255, 0, legend_h).astype(np.uint8)
                                legend = cv2.applyColorMap(legend.reshape(-1, 1), cv2.COLORMAP_JET)
                                legend = cv2.resize(legend, (legend_w, legend_h))

                                # Testo scala metri
                                step = legend_h // 5
                                for i, dist in enumerate([0, 10, 20, 30, 40, 50]):
                                    y = int(legend_h - (dist / 50.0) * legend_h)
                                    cv2.putText(legend, f"{dist}m", (2, max(12, y - 2)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                                # Affianca la legenda al frame depth
                                depth_for_display = np.hstack((depth_for_display, legend))
                            else:
                                depth_for_display = None

                            # Disegna bounding boxes YOLO
                            if bgr_for_display is not None and dets:
                                for _, bbox, _ in dets:
                                    cv2.rectangle(
                                        bgr_for_display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2
                                    )

                            # Mostra finestre
                            if bgr_for_display is not None:
                                cv2.imshow('RGB image', bgr_for_display)
                            if depth_for_display is not None:
                                cv2.imshow('Depth image', depth_for_display)

                            cv2.waitKey(1)

                    except Exception as e:
                        print(f"[DISPLAY] Error updating OpenCV windows: {e}")

                # Timeout per resettare stato
                if time.time() - mqtt_last_update > 0.7 and current_action != "normal":
                    current_action = "normal"

                vel = self.world.player.get_velocity()
                speed_mps = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                speed_kmh = speed_mps * 3.6
                
                if speed_kmh < 8 and current_action != "emergency_brake":
                    current_action="normal"

                # Eventi utente (solo se in stato 'normal') 
                if current_action == "normal":
                    if self.controller.parse_events(self.world, clock):
                        pass
                else:
                    pygame.event.pump()

                control = carla.VehicleControl()

                # Applica comportamento in base allo stato MQTT

                if current_action == "emergency_brake":
                    control.throttle = 0.0
                    control.brake = 1.0
                    self.world.player.apply_control(control)
                    print("[MQTT] Frenata di emergenza attiva")
                    continue

                elif current_action == "mild_brake":
                    control.throttle = 0.0
                    control.brake = level_brk if level_brk > 0 else 0.3
                    self.world.player.apply_control(control)
                    print(f"[MQTT] Frenata lieve applicata (level={control.brake:.2f})")
                    continue

                elif current_action == "warning":
                    print("[MQTT] Avviso al conducente: pedone vicino")

                # Render HUD e mondo 
                self.render(clock)

        finally:
            # Cleanup finale
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

    # Set window resolution
    args.res = '1280x720'
    args.width, args.height = CAMERA_WIDTH, CAMERA_HEIGHT #[int(x) for x in args.res.split('x')]

    # Set vehicle filter
    args.filter = 'vehicle.mercedes.coupe_2020'

    # Set synchronous mode
    args.sync = True

    log_level = mc.logging.DEBUG if args.debug else mc.logging.INFO
    mc.logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    mc.logging.info('listening to server %s:%s', args.host, args.port)

    return GameLoop(args)

##########################################
############ Simulation Setup ############
##########################################
existing_positions = []
if PEDESTRIAN == "DET":
    for _ in range(NUM_WALKERS):
        _, _, existing_positions = spawn_walker_det(world, existing_positions)
else:
    for _ in range(NUM_WALKERS):
        _, _ = spawn_walker(world)

# setup the simulation environment
game_loop = setup()

# get the vehicle and attach the camera
vehicle = world.get_actors().filter('vehicle.*')[0]
rgb_camera, depth_camera = setup_camera(vehicle)

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

threading.Thread(target=process_image, daemon=True).start()

def cleanup():
    remove_all(world)

try:
    game_loop.start()
except KeyboardInterrupt:
    cleanup()
finally:
    cleanup()