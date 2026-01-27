#!/bin/bash

BUILD_DIR=build-release
MPI_IMPL=UCX
TIME=8:00:0
EXP=rcs-weak
OUT=out/isc26/${EXP}/%j.out
PROG=examples/isc26/rcs

DEPTH=20
CYC=8
BLK=16
DIS_MAX=32

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

DIS_VALS="$(pow2s $DIS_MAX)"
for dis in $DIS_VALS; do
  TASKS=$(($dis * $dis))
  TPN=128

  NODES=$(($TASKS / $TPN))
  if [ $NODES -eq 0 ]; then
    NODES=1
    TPN=$TASKS
  fi

  ARGS="$DEPTH $CYC $dis $BLK SWAP QPD 0 x x"
  sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
    --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
    $QTNH_DIR/run/run.slurm
done
