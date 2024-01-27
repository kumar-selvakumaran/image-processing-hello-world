# Compiler
CC = g++

# Project Name
PROJECT = app

DATA = ../data/

DEST = ../bin/

# OpenCV Libraries (pkg-config)
LIBS = `pkg-config --cflags --libs opencv4` 

BINDIR = ../bin

# Compile and Link
TASK = vidDisplay

# TASK = imgDisplay


app: vidDisplay.cpp filter.cpp faceDetect.cpp
	$(CC) $^ -o $(BINDIR)/$(TASK) $(LIBS) -I../include -DENABLE_PRECOMPILED_HEADERS=OFF
	$(BINDIR)/$(TASK)

