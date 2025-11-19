#!/bin/bash

BUILD_DIR=build-release
MPI_IMPL=UCX
TIME=4:00:0
OUT=out/test/%j.out

# PROG=examples/test/bcmps-test
# sbatch -D $QTNH_DIR/run -N 1 -n 4 -o out/test/%j.out \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

#NROW=5,NCOL=5,DEPTH=8,CCHI=3,DCHI=2,LCHI=4
# PROG=examples/bench/mps-rcs
# ARGS="5 5 8 1 16 16"
# sbatch -D $QTNH_DIR/run -N 2 --ntasks-per-node=128 -o out/test/%j.out \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

# PROG=examples/bench/mps-rcs
# ARGS="5 5 8 4 16 4"
# sbatch -D $QTNH_DIR/run -N 2 --ntasks-per-node=128 -o out/test/%j.out \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

# PROG=examples/bench/mps-rcs
# ARGS="5 5 8 16 16 1"
# sbatch -D $QTNH_DIR/run -N 2 --ntasks-per-node=128 -o out/test/%j.out \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

# PROG=examples/bench/mps-rcs
# ARGS="5 5 8 2 16 8"
# sbatch -D $QTNH_DIR/run -N 2 --ntasks-per-node=128 -o out/test/%j.out \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

# PROG=examples/bench/mps-rcs
# ARGS="5 5 1 2 2 2"
# sbatch -D $QTNH_DIR/run -N 1 --ntasks-per-node=4 --time=$TIME -o $OUT \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

# PROG=examples/test/matrix-test
# ARGS=""
# sbatch -D $QTNH_DIR/run -N 1 --ntasks-per-node=4 --time=$TIME -o $OUT \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

PROG=examples/bench/mps-rcs
# ARGS="8 8 1 8 16 16 SVD"
# sbatch -D $QTNH_DIR/run -N 2 --ntasks-per-node=128 --time=$TIME -o $OUT \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

ARGS="9 6 4 8 64 16 QPD"
sbatch -D $QTNH_DIR/run -N 32 --ntasks-per-node=128 --time=$TIME -o $OUT \
  --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
  $QTNH_DIR/run/run.slurm
