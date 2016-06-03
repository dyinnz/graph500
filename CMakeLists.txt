project(isc-graph500)
cmake_minimum_required(VERSION 2.8)
find_package(MPI REQUIRED)
find_package (CUDA 7.5 REQUIRED)

include_directories(.)
include_directories(${MPI_INCLUDE_PATH})

file(GLOB Srcs ./src/*.cc)
file(GLOB CuSrcs "*.cu" "*.h")

cuda_compile(CUDA_O ${CuSrcs})

add_executable(graph500 ${Srcs} ${CUDA_O})
target_link_libraries(graph500 ${MPI_LIBRARIES})


set(FlagsOptimize "-O3 -march=native -mtune=native")
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -fopenmp -g")
