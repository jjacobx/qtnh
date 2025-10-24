#!/bin/bash

# Slurm job options (job-name, compute nodes, job time)
#SBATCH --job-name=qtnh
#SBATCH --time=0:40:0
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1

# Replace [budget code] below with your budget code (e.g. t01)
#SBATCH --account=d35-ja
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --output=out/ieee25/qftmps/%j.out

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
: ${EXP="ieee25-qftzmps"}
: ${PROG=qftmps}
: ${NSITES=4}
: ${CHIDIS=2}
: ${CHILOC_UP=32}
: ${CHILOC_DN=16} # If same as CHI_LOC_FR, don't compare. 
: ${BDIM=2}       # If 0, use |0> state. 
DIR=../build/examples/ieee25
EXE="$DIR/$PROG $NSITES $CHIDIS $CHILOC_UP $CHILOC_DN $BDIM"
MPI_ARGS="--distribution=block:block --hint=nomultithread"

# Launch the parallel job
echo "Experiment: ${EXP}"
echo "Running ${EXE}: "
srun $MPI_ARGS $EXE
echo "Finished running"
echo

# Wait for job stats
sleep 10

# Stats for pretty printing
PRETTY_STATS=jobid,jobname,state,exitcode,elapsed,consumedenergy,ReqCPUFreqMax

# Print useful job stats
echo "Job stats:"
sacct -j $SLURM_JOB_ID.0 --format=$PRETTY_STATS
echo

# Write all stats to file
sacct -j $SLURM_JOB_ID --json > out/jobinfo/$SLURM_JOB_ID.json