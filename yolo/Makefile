# adaptd from HOG Makefile, based on simpler-capture-4 Makefile
# https://github.com/sbsiewertcsu/cv4_computer_vision_starter_code/blob/main/simpler-capture-4/Makefile

INCLUDE_DIRS = -I/usr/include/opencv4 -I/usr/local/include/opencv4
LIB_DIRS =
CC=g++

CDEFS=
CFLAGS= -O0 -g $(INCLUDE_DIRS) $(CDEFS)
LIBS= -L/usr/lib -lopencv_core -lopencv_flann -lopencv_video -lrt

HFILES=
CFILES= yolo.cpp
SRCS= ${HFILES} ${CFILES}

all:	yolo

clean:
	-rm -f *.o *.d
	-rm -f yolo

yolo: yolo.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o `pkg-config --libs opencv4` $(LIBS)

depend:

yolo.o: yolo.cpp ../comTracker.hpp
	$(CC) $(CFLAGS) -c $< -o $@
