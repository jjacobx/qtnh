#!/bin/bash

BUILD_DIR=build-release
MPI_IMPL=UCX
TIME=8:00:0
OUT=out/isc26/riqft-block/%j.out
PROG=examples/isc26/riqft

SITES=40
DIS=16
LOC=256
SAT=1.0

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

CYC_VALS="$(pow2s $LOC)"
for cyc in $CYC_VALS; do
  BLK=$(($LOC / $cyc))
  ARGS="$SITES $cyc $DIS $BLK $SAT QPD"
  sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
    --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,EXP=$EXP,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
    $QTNH_DIR/run/run.slurm
done
