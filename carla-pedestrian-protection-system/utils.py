import math
import numpy as np
from typing import Tuple

def smooth_increase(level_brk, level_intensity, rate, speed):
    max_speed = 50.0
    speed_factor = min(speed / max_speed, 1.0)

    if speed < 10:
        delta = -100
    else:
        delta = rate * (level_intensity ** 1.2) * (speed_factor * 1.05)

    level_brk = max(0.0, level_brk + delta)  
    max_brk = 0.40 + 0.30 * speed_factor

    return min(level_brk, max_brk)


def max_yaw_allowed(distance):
    if distance <= 0:
        return 37.0
    elif distance >= 40.0:
        return 0.0
    else:
        return 37.0 * (1 - distance / 40.0)
    
def pixel_to_angle(u: int, v: int, K: np.ndarray) -> Tuple[float, float]:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) / fx
    y = (v - cy) / fy
    vec = np.array([1.0, x, -y])  
    vec /= np.linalg.norm(vec) 
    yaw = math.atan2(vec[1], vec[0])    
    pitch = math.atan2(vec[2], vec[0])  
    return yaw, pitch

def get_distance_to_pedestrian_centroid(centroid, depth_image):
    x, y = centroid
    blue  = depth_image[y, x, 0]
    green = depth_image[y, x, 1]
    red   = depth_image[y, x, 2]

    normalized_depth = (red + green * 256 + blue * 256**2) / (256**3 - 1)
    depth_in_meters = normalized_depth * 1000.0
    return depth_in_meters