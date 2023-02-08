#!/bin/bash

# this creates an environment with mpi4py from Delta's base anaconda3_gpu
module load gcc
module load openmpi
module load anaconda3_gpu
pip3 install --user mpi4py
