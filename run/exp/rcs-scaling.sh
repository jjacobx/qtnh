#!/bin/bash

sbatch -N 1   --ntasks-per-node=64 --export=NROW=5,NCOL=5,DEPTH=8,DCHI=8,LCHI=48,MPI_IMPL=UCX  run-mps-rcs.slurm
sbatch -N 4   --ntasks-per-node=64 --export=NROW=5,NCOL=5,DEPTH=8,DCHI=16,LCHI=24,MPI_IMPL=UCX run-mps-rcs.slurm
sbatch -N 9   --ntasks-per-node=64 --export=NROW=5,NCOL=5,DEPTH=8,DCHI=24,LCHI=16,MPI_IMPL=UCX run-mps-rcs.slurm
sbatch -N 16  --ntasks-per-node=64 --export=NROW=5,NCOL=5,DEPTH=8,DCHI=32,LCHI=12,MPI_IMPL=UCX run-mps-rcs.slurm
sbatch -N 36  --ntasks-per-node=64 --export=NROW=5,NCOL=5,DEPTH=8,DCHI=48,LCHI=8,MPI_IMPL=UCX  run-mps-rcs.slurm
#sbatch -N 64  --ntasks-per-node=64 --export=NROW=5,NCOL=5,DEPTH=8,DCHI=64,LCHI=6,MPI_IMPL=UCX  run-mps-rcs.slurm
#sbatch -N 144 --ntasks-per-node=64 --export=NROW=5,NCOL=5,DEPTH=8,DCHI=96,LCHI=4,MPI_IMPL=UCX  run-mps-rcs.slurm