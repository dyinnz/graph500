#!/bin/bash

thread=20

export CC=gcc
export CXX=g++

$CC -v
$CXX -v

case $1 in
  g | generate)
    rm -rf ./CMakeFiles ./CMakeCache.txt ./cmake_install.cmake
    cmake -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX .
    ;;
  *)
    make -j${thread} VERBOSE=1
    ;;
esac

