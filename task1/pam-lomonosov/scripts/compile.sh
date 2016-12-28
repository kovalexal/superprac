#!/bin/bash

mpic++ ../main.cpp -I/mnt/data/users/dm4/vol12/kovalexal_1854/boost_gcc_ompi/include -L/mnt/data/users/dm4/vol12/kovalexal_1854/boost_gcc_ompi/lib -lboost_mpi-mt -lboost_serialization-mt -std=c++11 -static-libstdc++ -O3 -o ../pam-lomonosov
