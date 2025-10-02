import carla
import cv2
import numpy as np
import pygame
import paho.mqtt.client as mqtt
from ultralytics import YOLO
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
import time
import threading
import math
from typing import Tuple
import requests
from Adas import Forward_collision_warning_mqtt, Fcw_state



#########################################
################ Classes ################
#########################################

@dataclass
class Pedestrian:
    x: int
    y: int
    distance: float # maybe int with fixed precision is enough
    time_to_collision: float
    yaw: float = 0.0
    pitch: float = 0.0

class Mode(Enum):
    KEYBOARD = 1
    STEERING_WHEEL = 2

##########################################
################# Config #################
##########################################

MODE = Mode.STEERING_WHEEL
CAMERA_DEBUG = True
NUM_WALKERS = 75

BROKER = "localhost"
PORT = 1883
TOPIC = "pedestrian_monitoring"

CAMERA_WIDTH = 2000
CAMERA_HEIGHT = 720
VIEW_FOV = 120


# radar_sensor = None
# input_radar = None
# input_radar_lock = threading.Lock()

model = YOLO("yolov8n.pt")
safety_state = "normal"   # può essere "normal", "soft", "emergency"


if CAMERA_DEBUG:
    cv2.namedWindow('RGB image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Depth image', cv2.WINDOW_NORMAL)

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
spectator = world.get_spectator()

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
    import manual_control_steeringwheel as mc
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

mqtt_client = mqtt.Client()

try:
    mqtt_client.connect(BROKER, PORT, 60)
    mqtt_client.loop_start()
    print(f"Listening to {TOPIC} on {BROKER}...")
except Exception as e:
    print("Connection failed:", e)

###########################################
############ Utility functions ############
###########################################
def get_asphalt_friction_coefficient():
    # Valori tipici: asciutto ~0.9, bagnato ~0.5, ghiaccio ~0.1
    return 0.9

def emergency_brake():
    global vehicle
    print("⚠️ EMERGENCY BRAKE TRIGGERED by FCW")
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
    # world -> camera
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
    rgb_bp.set_attribute('fov', str(VIEW_FOV))  # <-- aggiunto
    rgb_camera = world.spawn_actor(rgb_bp, camera_transform, attach_to=car)

    depth_bp = blueprint_library.find('sensor.camera.depth')
    depth_bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
    depth_bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
    depth_bp.set_attribute('fov', str(VIEW_FOV))  # <-- aggiunto
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
    """
    Spawns a pedestrian walker in the given CARLA world and makes it walk to a random destination.
    Args:
        world (carla.World): The CARLA world object where the walker will be spawned.
    Returns:
        tuple: A tuple containing the walker actor and its controller actor.
               If spawning fails, returns (None, None).
    """
    blueprint_library = world.get_blueprint_library()

    walker_blueprints = list(blueprint_library.filter('walker.pedestrian.*'))

    if not walker_blueprints:
        return None, None

    walker_bp = random.choice(walker_blueprints)

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
    if controller_bp is None:
        return walker, None

    controller = world.try_spawn_actor(controller_bp, carla.Transform(), walker)
    if controller:
        controller.start()
        # Make the walker walk randomly by continuously assigning new random destinations
        def random_walk():
            while controller.is_alive and walker.is_alive:
                destination = world.get_random_location_from_navigation()
                if destination:
                    controller.go_to_location(destination)
                controller.set_max_speed(speed)
                # Wait for a random time before assigning a new destination
                time.sleep(random.uniform(5, 15))
        threading.Thread(target=random_walk, daemon=True).start()

    return walker, controller


##########################################
############ Image Processing ############
##########################################

def detect_pedestrians(image):
    results = model.predict(image, device='cpu', verbose=False)[0] # device='cuda:0' for GPU inference
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

    # print(f"Depth at pixel ({x}, {y}): {depth_in_meters:.2f} meters")

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

def send_data_async(payload: dict):
    def _send():
        try:
            url = "http://localhost:5000/data" 
            requests.post(url, json=payload, timeout=1)
        except Exception as e:
            print("HTTP send failed:", e)
    threading.Thread(target=_send, daemon=True).start()


def process_image():
    """
    Process the input image:
      - legge RGB e Depth
      - fa YOLO per i pedoni
      - calcola distanza e time-to-collision
      - aggiorna safety_state
      - invia dati al server HTTP
      - aggiorna processed_output
    """
    global input_rgb_image, input_depth_image, processed_output, safety_state
    last_inference_time = 0.0
    target_dt = 0.10  # 2 Hz

    while True:
        # ---- acquisizione immagini ----
        with input_rgb_image_lock, input_depth_image_lock:
            if input_rgb_image is None or input_depth_image is None:
                continue
            rgb_image = input_rgb_image
            depth_image = input_depth_image

        now = time.time()
        if now - last_inference_time < target_dt:
            time.sleep(0.01)
            continue
        last_inference_time = now

        # ---- preprocessing immagini ----
        rgb_image = np.reshape(np.copy(rgb_image.raw_data), (rgb_image.height, rgb_image.width, 4))
        rgb_image = rgb_image[:, :, :3]
        depth_image = np.reshape(np.copy(depth_image.raw_data), (depth_image.height, depth_image.width, 4))

        # ---- velocità veicolo ----
        vehicle_speed = vehicle.get_velocity()
        vehicle_speed_mps = np.sqrt(vehicle_speed.x**2 + vehicle_speed.y**2 + vehicle_speed.z**2)

        # ---- detection pedoni (YOLO) ----
        detections = detect_pedestrians(rgb_image)
        detected_pedestrians: List[Pedestrian] = []
        for conf, _, centroid in detections:
            distance = get_distance_to_pedestrian_centroid(centroid, depth_image)
            time_to_collision = distance / vehicle_speed_mps if vehicle_speed_mps > 0 else float('inf')
            yaw, pitch = pixel_to_angle(centroid[0], centroid[1], rgb_camera.calibration)
            detected_pedestrians.append(
                Pedestrian(
                    x=centroid[0],
                    y=centroid[1],
                    distance=distance,
                    time_to_collision=time_to_collision,
                    yaw=yaw,
                    pitch=pitch
                )
            )

        # ---- trova pedone più vicino (camera) ----
        closest_ped = min(detected_pedestrians, key=lambda p: p.distance) if detected_pedestrians else None
        conf = max([c for c, _, _ in detections]) if detections else None
        yaw, pitch = (pixel_to_angle(closest_ped.x, closest_ped.y, rgb_camera.calibration)
                      if closest_ped else (None, None))
        ttc_camera = closest_ped.time_to_collision if closest_ped else None

        # ---- prepara payload ----
        payload = {
            "timestamp": now,
            "vehicle_speed": vehicle_speed_mps,
            "yolo_conf": float(conf) if conf else None,
            "camera_distance": closest_ped.distance if closest_ped else None,
            "camera_yaw_deg": math.degrees(yaw) if yaw is not None else None,
            "camera_pitch_deg": math.degrees(pitch) if pitch is not None else None,
            "camera_ttc": ttc_camera,
        }

        # ---- invio asincrono ----
        # send_data_async(payload)

        # ---- stampa riassuntiva ----
        # print("[SUMMARY]",
        #     # f"ts={now:.1f}",
        #     # f"speed={vehicle_speed_mps:.2f} m/s",
        #     f"YOLO conf={conf:.2f}" if conf else "YOLO none",
        #     f"cam_dist={closest_ped.distance:.1f}m" if closest_ped else "cam none",
        #     f"cam_ttc={ttc_camera:.2f}" if ttc_camera else "",
        #     f"cam_yaw={math.degrees(yaw):.1f}°" if yaw is not None else "",
        #     f"cam_pitch={math.degrees(pitch):.1f}°" if pitch is not None else "",
        # )

        # ---- salva output ----
        with processed_output_lock:
            processed_output = {
                "rgb_image": rgb_image,
                "depth_image": depth_image,
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
                print('WARNING: You are currently in asynchronous mode and could '
                    'experience some issues with the traffic simulation')

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
            while True:
                if self.args.sync:
                    self.sim_world.tick()
                clock.tick_busy_loop(self.fps)

                global safety_state

                # # ---- gestione frenata in base allo stato ----
                # if safety_state == "emergency":
                #     control = carla.VehicleControl(throttle=0.0, brake=1.0)
                #     self.world.player.apply_control(control)
                #     print("⚠️ EMERGENCY BRAKE - user input disabled")
                # elif safety_state == "soft":
                #     if self.controller.parse_events(self.world, clock):
                #         return
                #     control = self.world.player.get_control()
                #     control.throttle = min(control.throttle, 0.2)
                #     control.brake = max(control.brake, 0.3)
                #     self.world.player.apply_control(control)
                #     print("⚠️ Soft braking - user input limited")
                # else:
                if self.controller.parse_events(self.world, clock):
                    return

                try:
                    # ---- recupero output immagini e detections ----
                    with processed_output_lock:
                        output_rgb_image = processed_output["rgb_image"]
                        output_depth_image = processed_output["depth_image"]
                        output_detections = processed_output["detections"]


                    bgr_for_display = cv2.cvtColor(output_rgb_image, cv2.COLOR_RGB2BGR)
                    depth_for_display = cv2.cvtColor(output_depth_image, cv2.COLOR_RGB2BGR)

                    # ---- disegno pedoni YOLO ----
                    for _, bbox, _ in output_detections:
                        cv2.rectangle(bgr_for_display, (bbox[0], bbox[1]),
                                    (bbox[2], bbox[3]), (0, 255, 0), 2)

                    # ---- mostra le finestre ----
                    if CAMERA_DEBUG:
                        if bgr_for_display is not None:
                            cv2.imshow('RGB image', bgr_for_display)
                        if depth_for_display is not None:
                            cv2.imshow('Depth image', depth_for_display)

                except Exception:
                    pass

                self.render(clock)
                cv2.waitKey(1)
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

for _ in range(NUM_WALKERS):
    _, _ = spawn_walker(world)

# setup the simulation environment
game_loop = setup()

# get the vehicle and attach the camera
vehicle = world.get_actors().filter('vehicle.*')[0]
rgb_camera, depth_camera = setup_camera(vehicle)

fcw_system = Forward_collision_warning_mqtt(
    world=world,
    attached_vehicle=vehicle,
    get_asphalt_friction_coefficient=get_asphalt_friction_coefficient,
    action_listener=emergency_brake,
    mqtt_broker="localhost",   # puoi anche usare broker.emqx.io
    mqtt_port=1883
)

def rgb_camera_callback(image):
    try:
        global input_rgb_image
        with input_rgb_image_lock:
            input_rgb_image = image
    except Exception as e:
        # print(e.with_traceback())
        pass

def depth_camera_callback(image):
    try:
        global input_depth_image
        with input_depth_image_lock:
            input_depth_image = image
    except Exception as e:
        # print(e.with_traceback())
        pass

rgb_camera.listen(lambda image: rgb_camera_callback(image))
depth_camera.listen(lambda image: depth_camera_callback(image))

threading.Thread(target=process_image, daemon=True).start()

def cleanup():
    remove_all(world)
    try:
        fcw_system.destroy()
    except:
        pass

# start the game loop
try:
    game_loop.start()
except KeyboardInterrupt:
    cleanup()
finally:
    cleanup()