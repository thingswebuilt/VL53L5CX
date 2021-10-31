CC := gcc

# Set to TRUE to enable logging functions.
LOG_ENABLE = FALSE

CORE_INCLUDE_PATHS = -I../uld-driver/inc
PLATFORM_INCLUDE_PATHS = -I../platform
EXAMPLES_INCLUDE_PATHS = -I../examples

BASE_CFLAGS = -Wall -Wno-missing-braces
CFLAGS_RELEASE = -Os -g0
#CFLAGS_RELEASE += -DSTMVL53L5CX_KERNEL

LIB_CORE_SOURCES =\
	$(wildcard ../uld-driver/src/*.c)

LIB_PLATFORM_SOURCES =\
	$(wildcard ../platform/*.c)

LIB_EXAMPLES_SOURCES =\
	$(wildcard ../examples/*.c)

LIB_SOURCES := $(LIB_CORE_SOURCES) $(LIB_PLATFORM_SOURCES) $(LIB_EXAMPLES_SOURCES)
INCLUDE_PATH = $(CORE_INCLUDE_PATHS) $(PLATFORM_INCLUDE_PATHS) $(EXAMPLES_INCLUDE_PATHS)

CFLAGS = $(BASE_CFLAGS) $(CFLAGS_RELEASE) $(INCLUDE_PATH)

all:
	$(CC) $(CFLAGS) $(LIB_FLAGS) -shared -o menu.so -fPIC ./menu.c $(LIB_SOURCES)

clean:
	rm -f menu.so
