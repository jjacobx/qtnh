#!/bin/bash

BUILD_DIR=build-release
MPI_IMPL=UCX
TIME=8:00:0
OUT=out/isc26/rcs-block/%j.out
PROG=examples/isc26/rcs

DEPTH=4
DIS=16
LOC=256

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
  ARGS="$DEPTH $cyc $DIS $BLK SWAP QPD 0 x x"
  sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
    --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
    $QTNH_DIR/run/run.slurm
done
