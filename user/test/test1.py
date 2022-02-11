import ctypes

import sys
import signal

import numpy as np


sensor = ctypes.CDLL('/home/pi/diggity/ES_VL53L5CX/user/test/menu.so')


# # Sensor Init
status = sensor.init_sensor(0)
print(status)

# Attach a signal handler to catch SIGINT (Ctrl+C) and exit gracefully
def exit_handler(signal, frame):
    sensor.stop_ranging(0)
    print("Exiting Now")
    sys.exit(0)

signal.signal(signal.SIGINT, exit_handler)

# # Resolution Option
VL53L5CX_RESOLUTION_4X4 = ctypes.c_ulong(16)
VL53L5CX_RESOLUTION_8X8 = ctypes.c_ushort(64)

## Set Resolution
status = sensor.set_resolution(0,VL53L5CX_RESOLUTION_8X8)

# Set ranging frequency to 10Hz.
# Using 4x4, min frequency is 1Hz and max is 60Hz
# Using 8x8, min frequency is 1Hz and max is 15Hz

status = sensor.set_ranging_frequency(0,10)

# # VL53L5CX_TARGET Options

VL53L5CX_TARGET_ORDER_CLOSEST = ctypes.c_ushort(1)
VL53L5CX_TARGET_ORDER_STRONGEST	= ctypes.c_ushort(2)

status = sensor.set_target_order(0,VL53L5CX_TARGET_ORDER_CLOSEST)

## Get Integration Time

integration_time = sensor.get_integration_time_ms(0)
print("Integration Time ",integration_time)

status = sensor.start_ranging(0)

#status = sensor.check_data_ready(0)
global data
data = np.zeros((8, 8),dtype=int)


for reading in range(3):
    data = np.zeros((8,8),dtype=int)
    status = sensor.check_data_ready(0)

    counter1 = 0

    for x in range(8):
        for y in range(8):
            status = sensor.get_ranging_status(0,counter1)
            if(status == 5):
                tmp1 = sensor.get_ranging_data(0,counter1)
                data[x][y] = tmp1
            else:
                data[x][y] = -status

            counter1=counter1+1

    print(data)

