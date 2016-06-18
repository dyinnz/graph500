#!/bin/bash

/usr/lib64/openmpi/bin/mpirun -hostfile ./hostfile -npernode 20 ./run.sh $* > test.log
