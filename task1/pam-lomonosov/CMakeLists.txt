cmake_minimum_required(VERSION 3.6)
project(pam-lomonosov)

find_package(MPI REQUIRED)

set(CMAKE_C_COMPILER /usr/local/bin/mpicc)
set(CMAKE_CXX_COMPILER /usr/local/bin/mpicxx)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(Boost_INCLUDE_DIR /usr/local/Cellar/boost/1.62.0/include)
set(Boost_LIBRARY_DIR /usr/local/Cellar/boost/1.62.0/lib)
find_package(Boost COMPONENTS mpi serialization REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

set(SOURCE_FILES main.cpp)
add_executable(pam-lomonosov ${SOURCE_FILES})
target_link_libraries(pam-lomonosov mpi ${Boost_LIBRARIES})