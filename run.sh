#!/bin/bash

GCC_ROOT=/opt/gcc-4.9.3

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GCC_ROOT/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GCC_ROOT/lib64

./graph500 $*
