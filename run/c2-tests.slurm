#!/bin/bash

# Run this file with `sbatch -D $QTNH_DIR/run $QTNH_DIR/run/c2-tests.slurm`

# Slurm job options (job-name, compute nodes, job time)
#SBATCH --job-name=c2-tests
#SBATCH --time=0:20:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1

# Replace [budget code] below with your budget code (e.g. t01)
#SBATCH --account=d35-ja
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --output=out/c2/%j.out

# Set MPI implementation
: ${MPI_IMPL=OFI}
if [ "$MPI_IMPL" == "UCX" ]; then
  module load craype-network-ucx &> /dev/null
  module load cray-mpich-ucx &> /dev/null
fi

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
echo "MPI_IMPL=${MPI_IMPL}"
echo

# Print git info
echo "Git info: "
echo "Abbr. hash: $(git log -n 1 --format=%h)"
echo "Commit message: $(git log -n 1 --format=%s)"
echo "Commit date: $(git log -n 1 --format=%cs)"
echo "Branch: $(git branch --show-current)"
echo

# Set launch parameters
: ${BUILD_DIR=build}
EXE=../${BUILD_DIR}/tests/c2-mpi
MPI_ARGS="--distribution=block:block --hint=nomultithread"

# Launch parallel tests
echo "Running 2 rank tests: "
srun -n 2 $MPI_ARGS $EXE "[2rank]" --reporter root

echo "Running 3 rank tests: "
srun -n 3 $MPI_ARGS $EXE "[3rank]" --reporter root

echo "Running 4 rank tests: "
srun -n 4 $MPI_ARGS $EXE "[4rank]" --reporter root

echo "Running 6 rank tests: "
srun -n 6 $MPI_ARGS $EXE "[6rank]" --reporter root

echo "Running 8 rank tests: "
srun -n 8 $MPI_ARGS $EXE "[8rank]" --reporter root

echo "Running 16 rank tests: "
srun -n 16 $MPI_ARGS $EXE "[16rank]" --reporter root

echo "Running 32 rank tests: "
srun -n 32 $MPI_ARGS $EXE "[32rank]" --reporter root

echo "Finished running"
echo
