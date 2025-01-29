#!/bin/bash

# Slurm job options (job-name, compute nodes, job time)
#SBATCH --job-name=qtnh
#SBATCH --time=0:20:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1

#SBATCH --account=d35-ja
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --output=out/%j.out

# Load modules
module load forge

# Disable leak sanitier for MPI libs
export LSAN_OPTIONS=suppressions=lsan.supp,print_suppressions=0

# Set the number of threads to 1
export OMP_NUM_THREADS=1

# Propagate the cpus-per-task setting from script to srun commands
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# Print job environment
echo "Job environment: "
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME}"
echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES}"
echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo

# Set launch parameters
PROG=permute
NQUBITS=23
DQUBITS=3
NITER=1000
DIR=../build/examples/bench
EXE="$DIR/$PROG $NQUBITS $DQUBITS $NITER"
MPI_ARGS="--distribution=block:block --hint=nomultithread"
MAP_OUT="map/run-${SLURM_JOB_ID}_${PROG}_${SLURM_NTASKS}p_${NQUBITS}q_${DQUBITS}d_${NITER}i.map"

# Launch the parallel job
echo "Running and profiling ${EXE}: "
map -n $SLURM_NTASKS --mpi=slurm --mpiargs="$MPI_ARGS" -o "${MAP_OUT}.map" --profile $EXE
