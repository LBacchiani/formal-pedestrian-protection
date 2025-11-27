#!/usr/bin/env python3
import uinput
from evdev import InputDevice, ecodes

REAL_DEVICE = '/dev/input/event3'
dev = InputDevice(REAL_DEVICE)
print(f"🔧 Collegato al device reale: {dev.name}")

AXES_MAP = {
    0: uinput.ABS_X,   
    3: uinput.ABS_Y,   
    4: uinput.ABS_Z,   
    1: uinput.ABS_RZ, 
}

BUTTON_MAP = {
    288: uinput.BTN_TRIGGER,  
    289: uinput.BTN_THUMB,    
    290: uinput.BTN_TOP,     
    291: uinput.BTN_PINKIE,   
}

ABS_RANGE = (0, 65535, 0, 0)

events = (
    uinput.ABS_X + ABS_RANGE,
    uinput.ABS_Y + ABS_RANGE,
    uinput.ABS_Z + ABS_RANGE,
    uinput.ABS_RZ + ABS_RANGE,
) + tuple(BUTTON_MAP.values())

virtual = uinput.Device(events, name="VirtualWheel")
print("✅ Joystick virtuale creato come /dev/input/js1")

for event in dev.read_loop():
    if event.type == ecodes.EV_ABS and event.code in AXES_MAP:
        virtual.emit(AXES_MAP[event.code], event.value, syn=False)
        virtual.syn()
    elif event.type == ecodes.EV_KEY and event.code in BUTTON_MAP:
        virtual.emit(BUTTON_MAP[event.code], event.value)
