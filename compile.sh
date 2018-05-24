#!/bin/bash -e

g++ -I ./include src/main.cpp src/slic.cpp src/face_x.cpp src/fern.cpp src/regressor.cpp src/utils.cpp src/gc.cpp -o ./bin/main `pkg-config --cflags --libs opencv` -std=c++11
