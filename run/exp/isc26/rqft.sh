#!/bin/bash

BUILD_DIR=build-release
MPI_IMPL=UCX
TIME=2:00:0
OUT=out/isc26/rqft/%j.out
PROG=examples/isc26/rqft
EXP=rqft

SITES=40
CYC=2
DIS=16
BLK=4
RAND=2

TASKS=$(($DIS * $DIS))
TPN=128

NODES=$(($TASKS / $TPN))
if [ $NODES -eq 0 ]; then
  NODES=1
  TPN=$TASKS
fi

ARGS="$SITES $CYC $DIS $BLK $RAND QPD"
sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
  --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,EXP=$EXP,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
  $QTNH_DIR/run/run.slurm
