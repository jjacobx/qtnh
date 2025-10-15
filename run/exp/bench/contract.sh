#!/bin/bash

run_contract() {
  local dqs=$1  # distributed indices list
  local lqs=$2  # local indices list
  local mpi=$3  # MPI implementation
  local cpn=128 # cores per node

  local arr=($dqs)
  local d0=${arr[0]}
  local d1=${arr[1]}
  local d2=${arr[2]}

  [ $d0 -le $d1 ] && [ $d0 -le $d2 ] && local dq=$(($d1 + $d2))
  [ $d1 -le $d0 ] && [ $d1 -le $d2 ] && local dq=$(($d0 + $d2))
  [ $d2 -le $d0 ] && [ $d2 -le $d1 ] && local dq=$(($d0 + $d1))

  local n_proc=$((1 << $dq))
  local n_node=$((($n_proc - 1) / $cpn + 1))
  echo "n_proc=$n_proc,n_node=$n_node"

  local out=out/bench/%j.out
  local exp=bench-contract
  local prog=examples/bench/contract
  local args="$dqs $lqs 10"

  sbatch -D $QTNH_DIR/run -N $n_node -n $n_proc -o $out \
    --export=EXP=$exp,PROG=$prog,ARGS="$args",MPI_IMPL=$mpi \
    $QTNH_DIR/run/run.slurm
}

run_contract "4 4 4" "10 10 10" OFI
