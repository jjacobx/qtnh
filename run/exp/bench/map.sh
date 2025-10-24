#!/bin/bash

BUILD_DIR=build-map-release
MPI_IMPL=OFI

# PROG=examples/bench/contract
ARGS="8 0 0 19 1 1 10"
sbatch -D $QTNH_DIR/run -N 2 -n 256 \
  --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
  $QTNH_DIR/run/map.slurm

# PROG=examples/bench/contract
ARGS="4 4 4 10 10 10 10"
sbatch -D $QTNH_DIR/run -N 2 -n 256 \
  --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
  $QTNH_DIR/run/map.slurm

# PROG=examples/bench/decompose
ARGS="1 1 4 4 7 7 10"
sbatch -D $QTNH_DIR/run -N 2 -n 256 --time=1:00:0 \
  --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
  $QTNH_DIR/run/map.slurm

#NROW=5,NCOL=5,DEPTH=8,DCHI=16,LCHI=16
PROG=examples/bench/mps-rcs
ARGS="5 5 8 16 16"
sbatch -D $QTNH_DIR/run -N 2 --ntasks-per-node=128 --time=2:00:0 \
  --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
  $QTNH_DIR/run/map.slurm
