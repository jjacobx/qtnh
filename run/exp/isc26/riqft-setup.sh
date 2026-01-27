#!/bin/bash

BUILD_DIR=build-release
MPI_IMPL=UCX
TIME=8:00:0
OUT=out/isc26/riqft-setup/%j.out
PROG=examples/isc26/riqft

SITES=40
DIS=16
SAT=1.0

TASKS=$(($DIS * $DIS))
TPN=128

NODES=$(($TASKS / $TPN))
if [ $NODES -eq 0 ]; then
  NODES=1
  TPN=$TASKS
fi

IVALS="8 4 2 1"
DECS="SVD QPD"
for i in $IVALS; do
  for dec in $DECS; do
    ARGS="$SITES $i $DIS 16 $SAT $dec"
    sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
      --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,EXP=$EXP,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
      $QTNH_DIR/run/run.slurm

    ARGS="$SITES 1 $DIS $i $SAT $dec"
    sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
      --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,EXP=$EXP,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
      $QTNH_DIR/run/run.slurm
  done
done
