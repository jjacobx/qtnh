#!/bin/bash

BUILD_DIR=build-release
MPI_IMPL=UCX
TIME=2:00:0
OUT=out/test/sycamore/%j.out

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

# PROG=examples/bench/mps-rcs
# ARGS="9 6 4 4 16 16 QPD"
# sbatch -D $QTNH_DIR/run -N 2 --ntasks-per-node=128 --time=$TIME -o $OUT \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

# PROG=examples/ieee25/sycamore
# ARGS="8 16 64 16 SWAP QPD"
# sbatch -D $QTNH_DIR/run -N 32 --ntasks-per-node=128 --time=$TIME -o $OUT \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

# ARGS="8 16 16 16 MPO QPD"
# sbatch -D $QTNH_DIR/run -N 2 --ntasks-per-node=128 --time=$TIME -o $OUT \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

# BUILD_DIR=build-adios2
# PROG=examples/ieee25/sycamore
# ARGS="4 16 16 16 SWAP QPD 0 x io/rcs4-qpd4096.bp"
# sbatch -D $QTNH_DIR/run -N 2 --ntasks-per-node=128 --time=$TIME -o $OUT \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

# BUILD_DIR=build-adios2
# PROG=examples/ieee25/sycamore
# ARGS="4 2 16 16 SWAP SVD 0 x io/rcs4-svd512.bp"
# sbatch -D $QTNH_DIR/run -N 2 --ntasks-per-node=128 --time=$TIME -o $OUT \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

# BUILD_DIR=build-adios2
# PROG=examples/ieee25/sycamore
# ARGS="4 2 16 16 MPO QPD 0 x io/rcs4-mpo512.bp"
# sbatch -D $QTNH_DIR/run -N 2 --ntasks-per-node=128 --time=$TIME -o $OUT \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

# BUILD_DIR=build-adios2
# PROG=examples/ieee25/sycamore
# ARGS="4 2 16 2 SWAP QPD 0 x io/rcs4-qpd64.bp"
# sbatch -D $QTNH_DIR/run -N 2 --ntasks-per-node=128 --time=$TIME -o $OUT \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

# BUILD_DIR=build-adios2
# PROG=examples/test/overlap-test
# ARGS="rcs4-ref.bp io/rcs4-qpd256.bp"
# sbatch -D $QTNH_DIR/run -N 2 --ntasks-per-node=128 --time=$TIME -o $OUT \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

# BUILD_DIR=build-adios2
# PROG=examples/test/overlap-test
# ARGS="rcs4-ref.bp io/rcs4-qpd128.bp"
# sbatch -D $QTNH_DIR/run -N 2 --ntasks-per-node=128 --time=$TIME -o $OUT \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

BUILD_DIR=build-adios2
PROG=examples/test/overlap-test
ARGS="io/rcs4-qpd4096.bp io/rcs4-mpo512.bp"
sbatch -D $QTNH_DIR/run -N 2 --ntasks-per-node=128 --time=$TIME -o $OUT \
  --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
  $QTNH_DIR/run/run.slurm
