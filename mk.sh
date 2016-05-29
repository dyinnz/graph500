#!/bin/bash

eval `modulecmd sh load gcc/6.1.0-dy`
eval `modulecmd sh load intel/icc-16.0.3`


thread=20

export CC=icc
export CXX=icpc

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

