# Adatped from simpler-capture-4 Makefile
# https://github.com/sbsiewertcsu/cv4_computer_vision_starter_code/blob/main/simpler-capture-4/Makefile

INCLUDE_DIRS = -I/usr/include/opencv4 -I/usr/local/include/opencv4
LIB_DIRS = 
CC=g++

CDEFS=
CFLAGS= -O2 -g $(INCLUDE_DIRS) $(CDEFS)
LIBS= -L/usr/lib -lopencv_core -lopencv_flann -lopencv_video -lrt

HFILES= 
CFILES= hog.cpp
SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.cpp=.o}

all:	hog

clean:
	-rm -f *.o *.d
	-rm -f hog

hog: hog.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o `pkg-config --libs opencv4` $(LIBS)

depend:

hog.o: hog.cpp ../comTracker.hpp
	$(CC) $(CFLAGS) -c $< -o $@
