#!/bin/bash

BUILD_DIR=build-release
MPI_IMPL=UCX
TIME=2:00:0
OUT=out/isc26/accuracy/%j.out
PROG=examples/isc26/rcs

DEPTH=8
CYC=1
DIS=16
BLK=16

TASKS=$(($DIS * $DIS))
TPN=128

NODES=$(($TASKS / $TPN))
if [ $NODES -eq 0 ]; then
  NODES=1
  TPN=$TASKS
fi

# ms – MPO SVD
# mq – MPO QPD
# ss – SWAP SVD
# sq – SWAP QPD

ARGS="$DEPTH $CYC $DIS $BLK MPO SVD 0 x io/rcs${DEPTH}-ms${TASKS}.bp"
sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
  --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
  $QTNH_DIR/run/run.slurm

ARGS="$DEPTH $CYC $DIS $BLK MPO QPD 0 x io/rcs${DEPTH}-mq${TASKS}.bp"
sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
  --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
  $QTNH_DIR/run/run.slurm

ARGS="$DEPTH $CYC $DIS $BLK SWAP SVD 0 x io/rcs${DEPTH}-ss${TASKS}.bp"
sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
  --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
  $QTNH_DIR/run/run.slurm

ARGS="$DEPTH $CYC $DIS $BLK SWAP QPD 0 x io/rcs${DEPTH}-sq${TASKS}.bp"
sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
  --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
  $QTNH_DIR/run/run.slurm
