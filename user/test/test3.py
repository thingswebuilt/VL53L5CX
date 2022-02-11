import ctypes

import tvm

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

import dataloaders.transforms as transforms

iheight, iwidth = 480, 640  # raw image size
output_size = (228, 304)
to_tensor = transforms.ToTensor()

def val_transform(rgb):
    transform = transforms.Compose([
        transforms.Resize(240.0 / iheight),
        transforms.CenterCrop(output_size),
    ])
    rgb_np = transform(rgb)
    rgb_np = np.asfarray(rgb_np, dtype='float') / 255
    
    return rgb_np

def create_rgbd(rgb, sparse_depth):
    rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
    return rgbd

from tvm.contrib import graph_executor

dev = tvm.cuda(0)
dtype = "float32"
lib = tvm.runtime.load_module('rgbd8_trt16.so')
m = graph_executor.GraphModule(lib["default"](dev))


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

status = sensor.set_ranging_frequency(0,5)
if not (status):
    print("Ranging Frequency Set")

## VL53L5CX_TARGET Options

VL53L5CX_TARGET_ORDER_CLOSEST = ctypes.c_ushort(1)
VL53L5CX_TARGET_ORDER_STRONGEST	= ctypes.c_ushort(2)

status = sensor.set_target_order(0,VL53L5CX_TARGET_ORDER_STRONGEST)
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

data = np.zeros((8, 8),dtype=float)
data_sparse = np.zeros((228, 304),dtype=float)


ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)
#ax3 = plt.subplot(1,3,3)

im1 = ax1.imshow(frame)
#im2 = ax2.imshow(data , vmin=0, vmax=10)
im2 = ax2.imshow(data_sparse , interpolation = 'nearest', vmin=0, vmax=2)


plt.colorbar(im2,fraction=0.05, pad=0.04)


def init():
    im1.set_data(np.zeros((8, 8), dtype=int))
 

def animate(i):

    status = sensor.check_data_ready(0)
    counter1 = 0
    global data, frame, vs

    starttime1 = time.time()
    frame = vs.read()
    #frame = frame[:,40:,:]
    frame = imutils.resize(frame, width=304,height=228)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    print("Frame Shape",frame.shape)
   

    starttime = time.time()
    for x in range(8):
        for y in range(8):
            #print("X,Y",x,",",y)
            status = sensor.get_ranging_status(0,counter1)
            if(status == 5):
                tmp1 = sensor.get_ranging_data(0,counter1)
                if tmp1 < 1000:
                    data[x][y] = tmp1
                    data_sparse[int(x*28.5)][int(y*38)] = int(tmp1/100)
                else:
                    data[x][y] = 1000
                    data_sparse[int(x*28.5)][int(y*38)] = 10
            else:
                #data[x][y] = sensor.get_ranging_data(0,counter1)
                data[x][y] = 1000
                data_sparse[int(x*28.5)][int(y*38)] = 10
            


            counter1=counter1+1
    
    counter1 = 0
    
    input_name = "input"
    # rgb = val_transform(frame)

    # print("shape  ",rgb.shape," ",data_sparse.shape)

    rgbd = create_rgbd(frame, data_sparse)
    input_tensor = to_tensor(rgbd)
    input_tensor = input_tensor.unsqueeze(0)

    print("Setting input")
    m.set_input(input_name, input_tensor)
    print("Running")
    m.run()
    print("Getting output")
    # Get outputs
    tvm_output = m.get_output(0)
    tvm_output = np.squeeze(tvm_output.numpy())/100
    print(tvm_output.shape)
    print(np.max(tvm_output))
    print(np.min(tvm_output))
    from numpy import asarray
    from numpy import savetxt
    data2 = asarray(tvm_output)
    savetxt('data123.csv',data2,delimiter='.')

    im2.set_data(tvm_output)
    im1.set_data(frame)
    #im2.set_data(data)
    
anim = animation.FuncAnimation(plt.gcf(), animate, init_func=init)
plt.show()

    
