#!/bin/bash

BUILD_DIR=build-release
MPI_IMPL=UCX
TIME=24:00:0
OUT=out/isc26/rcs-setup/%j.out
PROG=examples/isc26/rcs

DEPTH=4
DIS=16

TASKS=$(($DIS * $DIS))
TPN=128

NODES=$(($TASKS / $TPN))
if [ $NODES -eq 0 ]; then
  NODES=1
  TPN=$TASKS
fi

IVALS="8 4 2 1"
INTS="MPO SWAP"
DECS="SVD QPD"
for i in $IVALS; do
  for int in $INTS; do
    for dec in $DECS; do
      ARGS="$DEPTH $i $DIS 16 $int $dec 0 x x"
      sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
        --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
        $QTNH_DIR/run/run.slurm

      ARGS="$DEPTH 1 $DIS $i $int $dec 0 x x"
      sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
        --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
        $QTNH_DIR/run/run.slurm
    done
  done
done
