import ctypes

import sys 
import signal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from imutils.video import VideoStream
import imutils
import time
import cv2

from multiprocessing import Process

# Camera Initialize
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Load Sensor Module
sensor = ctypes.CDLL('/home/ebad/VL53L5CX_Linux_driver_1.1.1/user/test/menu.so')

# # Sensor Init
status = sensor.init_sensor(0)
print(status)

# Attach a signal handler to catch SIGINT (Ctrl+C) and exit gracefully
def exit_handler(signal, frame):
    sensor.stop_ranging(0)
    vs.stop()
    cv2.destroyAllWindows()
    print("Exiting Now")
    sys.exit(0)

signal.signal(signal.SIGINT, exit_handler)

# # Resolution Option
VL53L5CX_RESOLUTION_4X4 = ctypes.c_ulong(16)
VL53L5CX_RESOLUTION_8X8 = ctypes.c_ushort(64)

## Set Resolution
status = sensor.set_resolution(0,VL53L5CX_RESOLUTION_8X8)
if not (status):
    print("Ranging Resolution Set")

# Set ranging frequency to 10Hz.
# Using 4x4, min frequency is 1Hz and max is 60Hz
# Using 8x8, min frequency is 1Hz and max is 15Hz

status = sensor.set_ranging_frequency(0,10)
if not (status):
    print("Ranging Frequency Set")

## VL53L5CX_TARGET Options

VL53L5CX_TARGET_ORDER_CLOSEST = ctypes.c_ushort(1)
VL53L5CX_TARGET_ORDER_STRONGEST	= ctypes.c_ushort(2)

status = sensor.set_target_order(0,VL53L5CX_TARGET_ORDER_CLOSEST)
if not (status):
    print("Target Order Set")

## Get Integration Time

integration_time = sensor.get_integration_time_ms(0)
print("Integration Time ",integration_time)

status = sensor.start_ranging(0)
if not (status):
    print("Ranging Started")


# ===== FRAME =========
## Initialize Data
frame = vs.read()
frame = imutils.resize(frame)
frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

data = np.zeros((8, 8),dtype=int)


ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)

im1 = ax1.imshow(frame)
im2 = ax2.imshow(data , vmin=0, vmax=1500)

plt.colorbar(im2,fraction=0.05, pad=0.04)


def init():
    im1.set_data(np.zeros((8, 8), dtype=int))

def animate(i):

    status = sensor.check_data_ready(0)
    counter1 = 0
    global data, frame, vs

    starttime1 = time.time()
    frame = vs.read()
    # frame = imutils.resize(frame, width=400,height=400)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
   

    starttime = time.time()
    for x in range(8):
        for y in range(8):
            status = sensor.get_ranging_status(0,counter1)
            if(status == 5):
                tmp1 = sensor.get_ranging_data(0,counter1)
                if tmp1 < 1600:
                    data[x][y] = tmp1
                else:
                    data[x][y] = 1600
            else:
                #data[x][y] = sensor.get_ranging_data(0,counter1)
                data[x][y] = 1600

            counter1=counter1+1
    
    counter1 = 0
    endtime = time.time()
    # data = np.fliplr(data)        
    im1.set_data(frame)
    im2.set_data(data)
    endtime2 = time.time()

    print(endtime-starttime," -- ",endtime2-starttime," --- ",endtime2-starttime1)
    
anim = animation.FuncAnimation(plt.gcf(), animate, init_func=init)
plt.show()

    
