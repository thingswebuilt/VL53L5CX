# STMICROELECTRONICS - VL53L5CX Linux driver
VL53L5CX Linux driver and test applications for linux

## Introduction
The proposed implementation is customized to run on a Jetson Nano, but can be adapted to run on any linux embedded platform,
as far as the VL53L5CX device is connected through I2C
Compile and run this driver in a full user mode, where the i2c commnication is handled with the /dev/i2c-1 file descriptor. This is the user mode

### compile the test examples, the platform adaptation layer and the uld driver
    $ nano vl53l5cx-uld-driver/user/test/Makefile
    Enable or disable the STMVL53L5CX_KERNEL cflags option depending on the wished uld driver mode : with a kernel module of fully in user.
    $ cd vl53l5cx-driver/user/test
    $ make
### compile the kernel module (kernel mode only)
    $ cd vl53l5cx-uld-driver/kernel
    $ make clean
    $ make
    $ sudo make insert
### run the test application menu
    $ cd vl53l5cx-uld-driver/user/test
    $ ./menu
