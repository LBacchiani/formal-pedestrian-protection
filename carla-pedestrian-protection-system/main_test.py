import carla, time, threading, random
from agents.navigation.basic_agent import BasicAgent

from SyncSimulation import SyncSimulation


client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Simulazione sensore umidit� dell'asflato
def get_asphalt_friction_coefficient():
    world = client.get_world()
    weather = world.get_weather()
    weather.precipitation_deposits
    weather.wetness
    return 0.8

def set_async(world):
    settings = world.get_settings()
    settings.synchronous_mode = False

    world.apply_settings(settings)
    world.tick()


def load_map(world, map_name):
    if not world.get_map().name == f"Carla/Maps/{map_name}":
        try:
            world = client.load_world(map_name)
        except RuntimeError:
            print("")


def destroy_all_vehicles(world):
    all_actors = world.get_actors()
    
    vehicles = all_actors.filter('vehicle.*')
    
    print(f"Numero di veicoli trovati: {len(vehicles)}")
    
    for vehicle in vehicles:
        vehicle.destroy()

def destroy_all_pedestrians(world):
    all_actors = world.get_actors()
    
    pedestrians = all_actors.filter('walker.pedestrian.*')
    
    print(f"Numero di pedoni trovati: {len(pedestrians)}")
    
    for pedestrian in pedestrians:
        pedestrian.destroy()

def disable_traffic_lights():
    traffic_lights = world.get_actors().filter('traffic.traffic_light')
    for traffic_light in traffic_lights:
        traffic_light.set_state(carla.TrafficLightState.Green)
        traffic_light.set_green_time(99999)
        traffic_light.freeze(True)


def spawn_vehicle(world, location, index, y = 0, color= '255,0,0'):
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*')[index]

    if vehicle_bp.has_attribute('color'):
        vehicle_bp.set_attribute('color', color)  # Rosso in formato RGB


    spawn_point = carla.Transform()
    spawn_point.location = location
    spawn_point.rotation = carla.Rotation(yaw = y)
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    world.wait_for_tick()
    return vehicle     

def actor_vehicle(world, vehicle, desitination, speed = 60):
    actor_agent = BasicAgent(vehicle)
    actor_agent.set_destination(desitination)
    actor_agent.set_target_speed(speed)
    actor_agent.ignore_vehicles()
    world.wait_for_tick()
    return actor_agent      

def spawn_walker(start_location, yaw = 0):
    walker_bp = random.choice(world.get_blueprint_library().filter("walker.pedestrian.*"))
    walker_bp.set_attribute('is_invincible', 'false')

    trans = carla.Transform()
    trans.location = start_location
    trans.rotation = carla.Rotation(yaw = yaw)

    return world.spawn_actor(walker_bp, trans)

def move_forward(vehicle, throttle):
    control = carla.VehicleControl()
    control.throttle = throttle
    vehicle.apply_control(control)

def move_agent(vehicle, actor_agent):
    vehicle.apply_control(actor_agent.run_step())

def move_forward_agent(vehicle, actor_agent,x,y):
    forward_location = vehicle.get_location() + carla.Location(x=x, y=y, z=0)  # Adjust 'x', 'y' based on vehicle's orientation
    actor_agent.set_destination(forward_location)
    vehicle.apply_control(actor_agent.run_step())

def stop_vehicle(vehicle):
    control = carla.VehicleControl()
    control.throttle = 0.0
    control.brake = 1.0
    control.hand_brake = True
    vehicle.apply_control(control)

def pedestrian_control(x=1,y =0,z = 0):
    walker_control = carla.WalkerControl()
    walker_control.speed = 1.3  # Velocit� in m/s (positivo in avanti)
    walker_control.direction = carla.Vector3D(x, y, z)
    return walker_control

def stop_pedestrian(walker):
    walker_control = carla.WalkerControl()
    walker_control.speed = 0  # Velocit� in m/s (positivo in avanti)
    walker.apply_control(walker_control)


def _move_pedestrian(walker, controller_w, sim, extra_time):
    while sim.run:
        walker.apply_control(controller_w)
        time.sleep(0.5)
    time.sleep(extra_time)
    stop_pedestrian(walker)

def move_pedestrian(walker, controller_w, sim, extra_time):
    threading.Thread(target=_move_pedestrian, args=(walker, controller_w, sim, extra_time,)).start()

def print_spectator_position():
    location = world.get_spectator().get_transform().location
    rotation = world.get_spectator().get_transform().rotation
    print(location)
    print(rotation)

def set_spectator_location(location, rotation):
    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(location, rotation))

def run_camera(camera, vehicle, sim):
    return threading.Thread(target=camera, args=(vehicle,sim))

def view_from_above(vehicle, sim):    
    while sim.run:
        distance_above = 40.0 
        spectator = world.get_spectator()
        vehicle_location = vehicle.get_location()
        
        
        spectator_location = carla.Location(
                x=vehicle_location.x, 
                y=vehicle_location.y,
                z=vehicle_location.z + distance_above
                )

        spectator_rotation = carla.Rotation(
            pitch=-90.0,
            yaw=0,
            roll=0.0
        )
        
        spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))

def view_from_behind(vehicle, sim):    
    while sim.run:
        distance_behind = 18.0 
        height = 5.0
        distance_to_the_side = 0
        
        spectator = world.get_spectator()
        
        vehicle_location = vehicle.get_location()
        vehicle_rotation = vehicle.get_transform().rotation
        
        spectator_location = carla.Location(
                x=vehicle_location.x - distance_behind * vehicle_rotation.get_forward_vector().x + distance_to_the_side * vehicle_rotation.get_right_vector().x,
                y=vehicle_location.y - distance_behind * vehicle_rotation.get_forward_vector().y + distance_to_the_side * vehicle_rotation.get_right_vector().y,
                z=vehicle_location.z + height
            )
    
        spectator.set_transform(carla.Transform(spectator_location, vehicle_rotation))
    
load_map(world, "Town10HD_Opt")  
disable_traffic_lights()  

start = carla.Location(x=-41.5, y=80, z=1)
vehicle_speed = 40
destination = carla.Location(x=-41.5, y=10, z=1)

pedestrian_position = carla.Location(x=-41.5, y=37, z=1)


destroy_all_vehicles(world)
destroy_all_pedestrians(world)

simulation = SyncSimulation(world)

vehicle = spawn_vehicle(world, start, target_vehicle_id, -90)
actor_agent = actor_vehicle(world, vehicle, destination, vehicle_speed)

walker = spawn_walker(pedestrian_position)

try:
    adas = attach_adas(simulation, world, vehicle, target_vehicle_ttc)
    simulation.run_sync_simulation()
    set_spectator_location(carla.Location(x=-37.506443, y=33.672970, z=1.671275),
        carla.Rotation(pitch=-6.648561, yaw=118.152756, roll=-0.000091))

    while simulation.run:
        move_agent(vehicle, actor_agent)
            
    stop_vehicle(vehicle)
    
except KeyboardInterrupt:
    print("simulation stopped")
    stop_vehicle(vehicle)
    simulation.stop_simulation()
finally:
    print("done")
    adas.destroy()