#!/bin/bash

BUILD_DIR=build-map-release
MPI_IMPL=OFI
TIME=2:00:0
OUT=map/bcmps/%j.out

# PROG=examples/bench/contract
ARGS="8 0 0 19 1 1 10"
sbatch -D $QTNH_DIR/run -N 2 -n 256 --time=$TIME -o $OUT \
  --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
  $QTNH_DIR/run/map.slurm

# PROG=examples/bench/contract
ARGS="4 4 4 10 10 10 10"
sbatch -D $QTNH_DIR/run -N 2 -n 256 --time=$TIME -o $OUT \
  --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
  $QTNH_DIR/run/map.slurm

# PROG=examples/bench/decompose
ARGS="1 1 4 4 7 7 10"
sbatch -D $QTNH_DIR/run -N 2 -n 256 --time=$TIME -o $OUT \
  --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
  $QTNH_DIR/run/map.slurm

#NROW=5,NCOL=5,DEPTH=8,DCHI=16,LCHI=16
PROG=examples/bench/mps-rcs
ARGS="5 5 8 1 16 16"
sbatch -D $QTNH_DIR/run -N 2 --ntasks-per-node=128 --time=$TIME -o $OUT \
  --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
  $QTNH_DIR/run/map.slurm

