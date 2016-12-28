#!/bin/sh

nvcc -rdc=true -arch=sm_20 -ccbin mpicxx *.cpp *.cu -O3 -o main.out
#mpicxx *.c -o main.out -O3
#nvcc hello.cu -o hello.out
