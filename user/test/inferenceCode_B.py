import time
import tvm
from tvm import relay

import h5py

import os
import numpy as np

import torch
import torchvision

import dataloaders.transforms as transforms

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



path1 = '/home/ebad/spadRGBD/data/nyudepthv2/train/cafe_0001a/00001.h5'

iheight, iwidth = 480, 640  # raw image size
output_size = (228, 304)

to_tensor = transforms.ToTensor()

def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb_image = np.array(h5f['rgb'])

    rgb_image = np.transpose(rgb_image, (1, 2, 0))
    depth = np.array(h5f['depth'])

    return rgb_image, depth

def dense_to_sparse(depth):
    """
    Samples pixels with `num_samples`/#pixels probability in `depth`.
    Only pixels with a maximum depth of `max_depth` are considered.
    If no `max_depth` is given, samples in all pixels
    """
    print("Points", points)
    vert = int(depth.shape[0] / points)
    horz = int(depth.shape[1] / points)

    sparse_array = np.zeros(depth.shape)

    for x in range(points):
        for y in range(points):
            # print(x,y)
            # sparse_array[int(((x+1)*vert)-1), int(((y + 1) * horz)-1)] = depth[int(((x + 1) * vert)-1), int(((y + 1) * horz)-1)]
            sparse_array[int(((x + 1) * vert) - vert/2), int(((y + 1) * horz) - horz/2)] = depth[int(((x + 1) * vert) - vert/2), int(((y + 1) * horz)/2)]

            
    return sparse_array

def val_transform(rgb, depth):
    depth_np = depth
    transform = transforms.Compose([
        transforms.Resize(240.0 / iheight),
        transforms.CenterCrop(output_size),
    ])
    rgb_np = transform(rgb)
    rgb_np = np.asfarray(rgb_np, dtype='float') / 255
    depth_np = transform(depth_np)

    return rgb_np, depth_np

def create_rgbd(rgb, depth):
    sparse_depth = dense_to_sparse(depth)
    rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
    return rgbd

points = 8
rgbd_mode = True


rgb1, depth1 = h5_loader(path1)
rgb2, depth2 = val_transform(rgb1, depth1)

depth_sparse = dense_to_sparse(depth2)

rgbd = create_rgbd(rgb2, depth2)
input_tensor = to_tensor(rgbd)
input_tensor = input_tensor.unsqueeze(0)

depth_tensor = to_tensor(depth_sparse)

plt.figure()
plt.imshow(depth_tensor,interpolation='nearest')
plt.show()

depth_tensor = depth_tensor.unsqueeze(0)
depth_tensor = depth_tensor.unsqueeze(0)

print("Input Tensor",input_tensor.shape)
print("Depth Tensor",depth_tensor.shape)


input_name = "input"

######################################################################################
# scripted_model = torch.jit.load("/home/ebad/spadRGBD/modelrgbd40.pt")


if rgbd_mode:
    shape_list = [(input_name, input_tensor.shape)]
else:
    shape_list = [(input_name, depth_tensor.shape)]


# mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

# from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt
# mod, config = partition_for_tensorrt(mod, params)

# target = "cuda"
# with tvm.transform.PassContext(opt_level=3, config={'relay.ext.tensorrt.options': config}):
#     lib = relay.build(mod, target=target, params=params)

# lib.export_library('rgbd40_trt.so')

############################################################################################

# ============================================================

# target = tvm.target.Target("cuda", host="llvm")

# with tvm.transform.PassContext(opt_level=3):
#     lib = relay.build(mod, target=target, params=params)

# ============================================================ 

from tvm.contrib import graph_executor

dev = tvm.cuda(0)

dtype = "float32"

lib = tvm.runtime.load_module('rgbd8_trt16.so')

m = graph_executor.GraphModule(lib["default"](dev))
# Set inputs

if rgbd_mode:
    m.set_input(input_name, input_tensor)
else:
    m.set_input(input_name, depth_tensor)

# Execute

m.run()

# Get outputs
tvm_output = m.get_output(0)

print("OP SHAPE",tvm_output.shape)

ftimer = m.module.time_evaluator("run", dev, repeat=10)
prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))


plt.figure()
plt.imshow(np.squeeze(tvm_output.numpy()),interpolation='nearest')
plt.show()
