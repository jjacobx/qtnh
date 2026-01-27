#!/bin/bash

BUILD_DIR=build-release
MPI_IMPL=UCX
TIME=24:00:0
OUT=out/isc26/fidelity-validation/%j.out
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

# Reference state. 
CHI=$((16 * $DIS * 16))
ARGS="$DEPTH 16 $DIS 16 SWAP QPD 0 x io/valid/rcs${DEPTH}-qpd${CHI}.bp"
sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
  --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
  $QTNH_DIR/run/run.slurm

IVALS="8 4 2 1"
for i in $IVALS; do
  CHI=$(($i * $DIS * 16))
  ARGS="$DEPTH $i $DIS 16 SWAP QPD 0 x io/valid/rcs${DEPTH}-qpd${CHI}.bp"
  sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
    --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
    $QTNH_DIR/run/run.slurm

  ARGS="$DEPTH $i $DIS 16 SWAP SVD 0 x io/valid/rcs${DEPTH}-svd${CHI}.bp"
  sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
    --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
    $QTNH_DIR/run/run.slurm

  CHI=$(($DIS * $i))
  ARGS="$DEPTH 1 $DIS $i SWAP QPD 0 x io/valid/rcs${DEPTH}-qpd${CHI}.bp"
  sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
    --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
    $QTNH_DIR/run/run.slurm

  ARGS="$DEPTH 1 $DIS $i SWAP SVD 0 x io/valid/rcs${DEPTH}-svd${CHI}.bp"
  sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
    --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
    $QTNH_DIR/run/run.slurm
done
