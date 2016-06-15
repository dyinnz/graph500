project(isc-graph500)
cmake_minimum_required(VERSION 2.8)
find_package(MPI REQUIRED)
find_package (CUDA 7.5 REQUIRED)

include_directories(.)
include_directories(${MPI_INCLUDE_PATH})
include_directories(${CUDA_INCLUDE})


file(GLOB Srcs ./src/*.cc)
file(GLOB CuSrcs ./src/*.cu ./src/*.cuh)

list(APPEND CUDA_NVCC_FLAGS "-std=c++11;")
cuda_compile(CUDA_O ${CuSrcs})

add_executable(graph500 ${Srcs} ${CUDA_O})
target_link_libraries(graph500 ${MPI_LIBRARIES})
target_link_libraries(graph500 ${CUDA_LIBRARIES})

set(FlagsOptimize "-O3 -march=native -mtune=native -fbuiltin -freciprocal-math -fno-trapping-math")
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -fopenmp ${FlagsOptimize}")
