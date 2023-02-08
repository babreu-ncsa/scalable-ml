#!/bin/bash
#SBATCH -J tf_16d4n
#SBATCH --mem=240g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64      # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4    # gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbka-delta-gpu
#SBATCH --time=12:00:00
#SBATCH --output=16d4n_a100_%j.out
### GPU options ###
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest

module purge
module load gcc
module load openmpi
module load anaconda3_gpu
module load cuda

# warm up devices
mpiexec -n 4 python3 mnist_scaling.py --n_epochs 1 --n_layers 1 --n_units 32 --batch_size 1024

# run
mpiexec -n 4 python3 mnist_scaling.py --n_epochs 10 --n_layers 12 --n_units 8192 --batch_size 256
