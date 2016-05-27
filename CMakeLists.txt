project(isc-graph500)
cmake_minimum_required(VERSION 2.8)

include_directories(.)

file(GLOB Srcs ./src/*.cc)

add_executable(graph500 ${Srcs})

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -g")
