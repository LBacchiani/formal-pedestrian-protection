import carla, math, random, paho.mqtt.client as mqtt
from enum import Enum

# Enum state del sistema adas
class Fcw_state(Enum):
    IDLE = 4
    WARNING = 3
    ACTION = 2
    ESCAPE = 1

class Forward_collision_warning_mqtt:
        
        __idle_counter = 0 

        def __init__(self,
                    world,
                    attached_vehicle,
                    get_asphalt_friction_coefficient,
                    action_listener, 
                    debug = True,
                    visual_debug_pixel_life_time = 0.25,  
                    visual_debug_max_point_number = 10, 
                    mqtt_broker = 'broker.emqx.io',
                    mqtt_port = 1883,
                    mqtt_topic = "carla/fcw_state",
                    min_fcw_state = Fcw_state.IDLE,
                    vehicle_half_vertical_dimension = 0.75,
                    vehicle_half_horizontal_dimension = 0.9,
                    vehicle_wheelbase = 2.8,
                    min_ttc = 0.5,
                    average_reaction_time = 2.5,
                    velocity_th = 2.77,
                    escape_ratio_th = 0.5,
                    max_radiant_steer_angle = 1.22, # circa 70 gradi
                    steer_tollerance = 0.02,
                    radar_range = 285,
                    radar_height = 0.9,
                    climb_inconsistencies_height_th = 0.2,
                    max_slope = 0.2,
                    detected_point_th = 25,   
                    idle_counter_th = 30
        ):
            
            # Inizializzazione parametri
            self.__world = world
            self.__attached_vehicle = attached_vehicle
            self.__get_asphalt_friction_coefficient = get_asphalt_friction_coefficient
            self.__action_listener = action_listener
            self.__debug = debug
            self.__visual_debug_pixel_life_time = visual_debug_pixel_life_time
            self.__visual_debug_max_point_number = visual_debug_max_point_number
            self.__mqtt_topic = mqtt_topic
            self.__min_fcw_state = min_fcw_state
            self.__vehicle_half_vertical_dimension = vehicle_half_vertical_dimension
            self.__vehicle_half_horizontal_dimension = vehicle_half_horizontal_dimension
            self.__vehicle_wheelbase = vehicle_wheelbase
            self.__min_ttc = min_ttc
            self.__average_reaction_time = average_reaction_time
            self.__velocity_th = velocity_th
            self.__escape_ratio_th = escape_ratio_th
            self.__max_radiant_steer_angle = max_radiant_steer_angle
            self.__radar_range = radar_range
            self.__radar_height = radar_height
            self.__steer_tollerance = steer_tollerance
            self.__climb_inconsistencies_height_th = climb_inconsistencies_height_th
            self.__detected_point_th = detected_point_th
            self.__max_slope = max_slope
            self.__idle_counter_th = idle_counter_th

            # Creazione client mqtt
            self.__mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            self.__mqttc.connect(mqtt_broker, mqtt_port)

            # Creazione del radar
            rad_bp = world.get_blueprint_library().find('sensor.other.radar')
            rad_bp.set_attribute('horizontal_fov', str(25))
            rad_bp.set_attribute('vertical_fov', str(25))
            rad_bp.set_attribute('range', str(self.__radar_range))
            rad_bp.set_attribute('points_per_second', str(50000))
            rad_bp.set_attribute('sensor_tick', str(0.05))
            rad_location = carla.Location(x=2.25, z=self.__radar_height)
            rad_rotation = carla.Rotation()
            rad_transform = carla.Transform(rad_location,rad_rotation)

            # Aggancio al veicolo
            self.__radar = world.spawn_actor(rad_bp,rad_transform,attach_to=self.__attached_vehicle, attachment_type=carla.AttachmentType.Rigid)

            # Avvio client mqtt
            self.__mqttc.loop_start()

            # Settaggio del listener
            self.__radar.listen(lambda radar_data: self.__forward_collision_callback(radar_data, self.__attached_vehicle.get_control(), self.__get_attached_vehicle_velocity()))
                
        # Algoritmo
        def __forward_collision_callback(self, radar_data, attached_vehicle_control, attached_vehicle_velocity):
            asphalt_friction_coefficient = self.__get_asphalt_friction_coefficient()
            asphalt_friction_deceleration = 9.81 * asphalt_friction_coefficient
            if -self.__steer_tollerance < attached_vehicle_control.steer < self.__steer_tollerance:
                radiant_steer_angle = 0 
            else:
               radiant_steer_angle = attached_vehicle_control.steer * self.__max_radiant_steer_angle
            attached_vehicle_stimated_velocity, filtered_radar_data = self.__get_attached_vehicle_stimated_velocity_and_filtered_radar_data(radar_data, radiant_steer_angle)
            detected_escape_list = []
            detected_action_list = []
            detected_warning_list = []
            detected_idle_list = []
            if attached_vehicle_velocity > self.__velocity_th:
                for detection in filtered_radar_data:
                    projected_depth = self.__get_projected_depth(detection.azimuth, detection.altitude, detection.depth, radiant_steer_angle)
                    max_depth = self.__get_max_depth(detection.azimuth, radiant_steer_angle)
                    if projected_depth <= max_depth:
                        projected_velocity = self.__get_projected_velocity(detection.azimuth, detection.altitude, detection.velocity, radiant_steer_angle) 
                        breaking_distance = self.__get_breaking_distance(projected_velocity, asphalt_friction_deceleration)
                        reacting_distance = projected_depth - breaking_distance
                        ttc = reacting_distance / projected_velocity
                        if ttc < self.__min_ttc:  
                            if attached_vehicle_stimated_velocity < projected_velocity * self.__escape_ratio_th:
                                detected_escape_list.append((detection, projected_depth))
                            else:
                                detected_action_list.append((detection, projected_depth)) 
                        elif ttc < self.__average_reaction_time:
                            detected_warning_list.append((detection, projected_depth))
                        else:
                            detected_idle_list.append((detection, projected_depth))
                self.__analize_detection(detected_escape_list, radar_data, Fcw_state.ESCAPE, 0, 0, 1) 
                self.__analize_detection(detected_action_list, radar_data, Fcw_state.ACTION, 1, 0, 0, self.__action_listener)
                self.__analize_detection(detected_warning_list, radar_data, Fcw_state.WARNING, 1, 1, 0)
                self.__analize_detection(detected_idle_list, radar_data, Fcw_state.IDLE, 0, 1, 0)  

                if not (len(detected_escape_list) > 0 or len(detected_action_list) > 0 or len(detected_warning_list) > 0):
                    self.__idle_counter += 1
                    if self.__idle_counter >= self.__idle_counter_th:
                        self.__min_fcw_state = Fcw_state.IDLE
                else:
                    self.__idle_counter = 0

            else:
                self.__min_fcw_state = Fcw_state.IDLE
                self.__idle_counter = 0

        # Support 
        def __analize_detection(self, detected_point_list, radar_data, fwd_state, red, green, blue, listener = lambda: None):
            if len(detected_point_list) > self.__detected_point_th:
                if self.__check_climb(detected_point_list):
                    red, green, blue = 1,1,1
                elif fwd_state.value < self.__min_fcw_state.value:
                    self.__min_fcw_state = fwd_state
                    if fwd_state != Fcw_state.IDLE:
                        print(f"[FCW] Stato cambiato: {fwd_state.name}")  # <-- PRINT AGGIUNTA
                        self.__publish_message(fwd_state.name)
                    listener()
                number_of_point_to_display = min(len(detected_point_list), self.__visual_debug_max_point_number)
                sampled_detected_point_list = random.sample(detected_point_list, number_of_point_to_display)
                for detection_plus_projected_distance in sampled_detected_point_list:
                    self.__radar_visual_debug(detection_plus_projected_distance[0], radar_data, red, green, blue)

        # Debug visivo
        def __radar_visual_debug(self, detection, radar_data, red, green, blue):
            if self.__debug:
                current_rot = radar_data.transform.rotation
                azi = math.degrees(detection.azimuth)
                alt = math.degrees(detection.altitude)
                
                fw_vec = carla.Vector3D(x=detection.depth - 0.25)
                carla.Transform(
                    carla.Location(),
                    carla.Rotation(
                        pitch=current_rot.pitch + alt,
                        yaw=current_rot.yaw + azi,
                        roll=current_rot.roll)).transform(fw_vec)
                
                self.__world.debug.draw_point(
                    radar_data.transform.location + fw_vec,
                    size=0.075,
                    life_time=self.__visual_debug_pixel_life_time,
                    persistent_lines=False,
                    color=carla.Color(red, green, blue))

        # Controllori
        def __are_sign_equals(self, a, b):
            return (a > 0 and b > 0) or (a < 0 and b < 0)

        def __check_horizontal_collision(self, azimuth, depth, radiant_steer_angle):
            return abs(depth * math.sin(azimuth - radiant_steer_angle)) < self.__vehicle_half_horizontal_dimension

        def __check_vertical_collision(self, altitude, depth):
            return abs(depth * math.sin(altitude)) < self.__vehicle_half_vertical_dimension 

        def __check_climb(self, detections_plus_projected_distance):
            if len(detections_plus_projected_distance) > 2:
                climb_inconsistencies_height_counter = 0
                for i in range(2, len(detections_plus_projected_distance)):
                    climb_inconsistencies_height_1 = self.__get_climb_inconsistencies_height(detections_plus_projected_distance[i], detections_plus_projected_distance[i-1])
                    if climb_inconsistencies_height_1 != 0:
                        climb_inconsistencies_height_2 = self.__get_climb_inconsistencies_height(detections_plus_projected_distance[i], detections_plus_projected_distance[i-2])
                        if climb_inconsistencies_height_2 != 0:
                            climb_inconsistencies_height_counter += min(climb_inconsistencies_height_1, climb_inconsistencies_height_2)
                            if climb_inconsistencies_height_counter > self.__climb_inconsistencies_height_th:
                                return False
                        else: 
                            climb_inconsistencies_height_counter = 0
                    else:
                        climb_inconsistencies_height_counter = 0  
                return True
            return False
        
        # Convertitori
        def __get_projected_velocity(self, azimuth, altitude, velocity, radiant_steer_angle):
            return abs(math.cos(azimuth - radiant_steer_angle) * math.cos(altitude) * velocity)

        def __get_projected_depth(self, azimuth, altitude, depth, radiant_steer_angle):
            return abs(math.cos(azimuth - radiant_steer_angle) * math.cos(altitude) * depth)

        def __get_breaking_distance(self, projected_velocity, asphalt_friction_deceleration):
            return (1/2) * (projected_velocity**2) / asphalt_friction_deceleration
        

        def __get_attached_vehicle_velocity(self):
            velocity_vector = self.__attached_vehicle.get_velocity()
            velocity = (velocity_vector.x**2 + velocity_vector.y**2 + velocity_vector.z**2)**0.5
            if self.__attached_vehicle.get_control().reverse:
                velocity = -velocity
            return velocity

        def __get_attached_vehicle_stimated_velocity_and_filtered_radar_data(self, radar_data, radiant_steer_angle):
            velocity_sum = 0
            filtered_radar_data = []
            for detection in radar_data:
                velocity_sum += self.__get_projected_velocity(detection.azimuth, detection.altitude, detection.velocity, radiant_steer_angle)
                if detection.velocity < 0 and self.__check_horizontal_collision(detection.azimuth, detection.depth, radiant_steer_angle) and self.__check_vertical_collision(detection.altitude, detection.depth):
                    filtered_radar_data.append(detection)
            attached_vehicle_stimated_velocity = velocity_sum / len(radar_data)
            return attached_vehicle_stimated_velocity, filtered_radar_data

        def __get_max_depth(self, azimuth, radiant_steer_angle):
            if radiant_steer_angle == 0:
                return self.__radar_range
            radius = abs(self.__vehicle_wheelbase / math.tan(radiant_steer_angle))
            if self.__are_sign_equals(azimuth, radiant_steer_angle):
                cathetus = radius - 2 * self.__vehicle_half_horizontal_dimension
            else:
                cathetus = radius - self.__vehicle_half_horizontal_dimension
            stering_range = math.sqrt(radius**2 - cathetus**2)
            return min(stering_range, self.__radar_range)

        def __get_climb_inconsistencies_height(self, detection_plus_projected_distance1, detection_plus_projected_distance2):
            detection1 = detection_plus_projected_distance1[0]
            detection2 = detection_plus_projected_distance2[0]
            height1 = detection1.depth * math.sin(detection1.altitude) + self.__radar_height
            height2 = detection2.depth * math.sin(detection2.altitude) + self.__radar_height
            vertical_difference = height1 - height2
            horizontal_difference = detection_plus_projected_distance1[1] - detection_plus_projected_distance2[1]
            slope = vertical_difference / horizontal_difference
            if horizontal_difference > 0 and slope < self.__max_slope:
                return 0
            return vertical_difference

        # Invio messaggi Mqtt
        def __publish_message(self, message):
            status = self.__mqttc.publish(self.__mqtt_topic, message)
            if status[0] == 0:
                print(f"Send `{message}` to topic `{self.__mqtt_topic}`")
            else:
                print(f"Failed to send message to topic {self.__mqtt_topic}")

        # Destroy
        def destroy(self):
            self.__mqttc.loop_stop()
            self.__mqttc.disconnect()
            self.__radar.stop()
            self.__radar.destroy()