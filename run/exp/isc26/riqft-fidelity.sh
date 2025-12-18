#!/bin/bash

BUILD_DIR=build-release
MPI_IMPL=UCX
TIME=12:00:0
EXP=riqft-fidelity
OUT=out/isc26/${EXP}/%j.out
PROG=examples/isc26/riqft

DIS=8
BLK=16

TASKS=$(($DIS * $DIS))
TPN=128

NODES=$(($TASKS / $TPN))
if [ $NODES -eq 0 ]; then
  NODES=1
  TPN=$TASKS
fi

pow2s() {
  local max=$1
  local i=1

  while :; do
    echo -n $i
    i=$((2 * $i))

    if [ $i -gt $max ]; then
      break
    fi

    echo -n " "
  done
}

# Sites: 40, 100
# Chis: 512, 1024
# Sat: 1/1, 1/2, 1/4, 1/8, 1/16

SITES_VALS="40 100"
CYC_VALS="4 8"
DIV_VALS="$(pow2s 16)"
for sites in $SITES_VALS; do
  for cyc in $CYC_VALS; do
    for div in $DIV_VALS; do
      SAT=$(bc <<< "scale=2; 1/$div")

      ARGS="$sites $cyc $DIS $BLK $SAT QPD"
      sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
        --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,EXP=$EXP,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
        $QTNH_DIR/run/run.slurm
    done
  done
done
