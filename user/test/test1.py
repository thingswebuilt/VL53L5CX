import ctypes

import sys 
import signal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sensor = ctypes.CDLL('/home/ebad/VL53L5CX_Linux_driver_1.1.1/user/test/menu.so')
counter2 = 0
data2 = np.zeros((8, 8),dtype=int)

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
print(status)

# # VL53L5CX_TARGET Options

VL53L5CX_TARGET_ORDER_CLOSEST = ctypes.c_ushort(1)
VL53L5CX_TARGET_ORDER_STRONGEST	= ctypes.c_ushort(2)

status = sensor.set_target_order(0,VL53L5CX_TARGET_ORDER_CLOSEST)
print(status)

## Get Integration Time

integration_time = sensor.get_integration_time_ms(0)
print("Integration Time ",integration_time)

status = sensor.start_ranging(0)
print(status)

#status = sensor.check_data_ready(0)

data = np.zeros((8, 8),dtype=int)

fig = plt.figure()
im = plt.imshow(data , vmin=0, vmax=2000)
plt.colorbar(fraction=0.1, pad=0.04)


def init():
    im.set_data(np.zeros((8, 8), dtype=int))

def animate(i):

    status = sensor.check_data_ready(0)
    counter1 = 0
    global data

    for x in range(8):
        for y in range(8):
            status = sensor.get_ranging_status(0,counter1)
            if(status == 5):
                tmp1 = sensor.get_ranging_data(0,counter1)
                if tmp1 < 2000:
                    data[x][y] = tmp1
                else:
                    data[x][y] = 2000
            else:
                #data[x][y] = sensor.get_ranging_data(0,counter1)
                data[x][y] = 2000

            counter1=counter1+1

    counter1 = 0
    global counter2, data2
    if counter2 < 50:
        data2 = data2 + data
    counter2 = counter2 + 1
    # data = np.fliplr(data)        
    print(data2)
    im.set_data(data)

anim = animation.FuncAnimation(fig, animate, init_func=init)
plt.show()

    
