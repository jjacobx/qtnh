#!/bin/bash

BUILD_DIR=build-release
MPI_IMPL=UCX
TIME=1:00:0
EXP=fidelity-overlaps
OUT=out/isc26/${EXP}/%j.out
PROG=examples/test/overlap-test

DEPTH=4
DIS=16

TASKS=$(($DIS * $DIS))
TPN=128

NODES=$(($TASKS / $TPN))
if [ $NODES -eq 0 ]; then
  NODES=1
  TPN=$TASKS
fi

IVALS="16 32 64 128 256 512 1024 2048"
for i in $IVALS; do
  ARGS="io/valid/rcs4-qpd4096.bp io/valid/rcs${DEPTH}-qpd${i}.bp"
  sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
    --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
    $QTNH_DIR/run/run.slurm

  ARGS="io/valid/rcs4-qpd4096.bp io/valid/rcs${DEPTH}-svd${i}.bp"
  sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
    --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
    $QTNH_DIR/run/run.slurm
done
