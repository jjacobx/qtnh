#!/bin/bash

BUILD_DIR=build-profile-ofi
MPI_IMPL=OFI
TIME=48:00:0
OUT=out/isc26/rcs-profiling/%j.out
PROG=examples/isc26/rcs
EXP=rcs-profiling

DEPTH=4
CYC=16
DIS=16
BLK=16

TASKS=$(($DIS * $DIS))
TPN=128

NODES=$(($TASKS / $TPN))
if [ $NODES -eq 0 ]; then
  NODES=1
  TPN=$TASKS
fi

# ARGS="$DEPTH $CYC $DIS $BLK SWAP QPD 0 x x"
# sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL,EXP=$EXP \
#   $QTNH_DIR/run/map.slurm

# BUILD_DIR=build-release
# ARGS="1 16 $DIS 16 SWAP QPD 0 x x"
# sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL,EXP=$EXP \
#   $QTNH_DIR/run/run.slurm

# ARGS="1 32 16 16 SWAP QPD 0 x x"
# sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL,EXP=$EXP \
#   $QTNH_DIR/run/run.slurm

QOS=standard
TIME=24:00:0
ARGS="$DEPTH $CYC $DIS $BLK SWAP SVD 0 x x"
sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT --qos=$QOS \
  --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL,EXP=$EXP \
  $QTNH_DIR/run/map.slurm
