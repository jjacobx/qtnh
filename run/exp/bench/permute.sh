#!/bin/bash

run_permute() {
  local nq=$1   # total qubits
  local dq=$2   # distributed qubits
  local mpi=$3  # MPI implementation
  local cpn=128 # cores per node

  local n_proc=$((1 << $dq))
  local n_node=$((($n_proc - 1) / $cpn + 1))
  echo "n_proc=$n_proc,n_node=$n_node"

  local out=out/bench/%j.out
  local exp=bench-permute
  local prog=examples/bench/permute
  local args="$nq $dq 100"

  sbatch -D $QTNH_DIR/run -N $n_node -n $n_proc -o $out \
    --export=EXP=$exp,PROG=$prog,ARGS="$args",MPI_IMPL=$mpi \
    $QTNH_DIR/run/run.slurm
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
