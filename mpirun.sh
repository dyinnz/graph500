#!/bin/bash

/opt/openmpi-1.10.2/bin/mpirun -hostfile ./hostfile -npernode 4 ./run.sh $* > test.log
