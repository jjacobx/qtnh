#!/bin/bash

run_permute() {
  local nq=$1   # total qubits
  local dq=$2   # distributed qubits
  local mpi=$3  # MPI implementation
  local cpn=128 # cores per node

  local n_proc=$((1 << $dq))
  local n_node=$((($n_proc - 1) / $cpn + 1))
  echo "n_proc=$n_proc,n_node=$n_node"

  sbatch -D $QTNH_DIR/run -N $n_node -n $n_proc \
    --export=PROG=permute,NQ=$nq,DQ=$dq,IT=100,MPI_IMPL=$mpi \
    $QTNH_DIR/run/run-bench.slurm
}

DQS=$(seq 0 10)

for dq in ${DQS}; do
  wait_running_jobs 40
  run_permute $(($dq + 20)) $dq OFI
done

for dq in ${DQS}; do
  wait_running_jobs 40
  run_permute $(($dq + 22)) $dq OFI
done

for dq in ${DQS}; do
  wait_running_jobs 40
  run_permute $(($dq + 24)) $dq OFI
done

for dq in ${DQS}; do
  wait_running_jobs 40
  run_permute $(($dq + 20)) $dq UCX
done

for dq in ${DQS}; do
  wait_running_jobs 40
  run_permute $(($dq + 22)) $dq UCX
done

for dq in ${DQS}; do
  wait_running_jobs 40
  run_permute $(($dq + 24)) $dq UCX
done
