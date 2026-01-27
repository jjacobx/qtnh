#!/bin/bash

BUILD_DIR=build-release
MPI_IMPL=UCX
TIME=5:00:0
EXP=rcs
OUT=out/isc26/${EXP}/%j.out
PROG=examples/isc26/rcs

DEPTH=20
CYC=4
DIS=64
BLK=16

TASKS=$(($DIS * $DIS))
TPN=128

NODES=$(($TASKS / $TPN))
if [ $NODES -eq 0 ]; then
  NODES=1
  TPN=$TASKS
fi

#FILE_OUT=io/isc26/rcs20-qpd8k.bp
FILE_OUT=x
ARGS="$DEPTH $CYC $DIS $BLK SWAP QPD 0 x ${FILE_OUT}"
sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
  --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
  $QTNH_DIR/run/run.slurm

# ARGS="1 $CYC $DIS $BLK SWAP QPD 0 x io/isc26/rcs4-qpd16k.bp"
# sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

# ARGS="1 $CYC $DIS $BLK SWAP QPD 1 io/isc26/rcs1-qpd16k.bp io/isc26/rcs4-qpd16k.bp"
# sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

# ARGS="2 $CYC $DIS $BLK SWAP QPD 2 io/isc26/rcs2-qpd16k.bp io/isc26/rcs4-qpd16k.bp"
# sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

# ARGS="4 $CYC $DIS $BLK SWAP QPD 4 io/isc26/rcs4-qpd16k.bp io/isc26/rcs8-qpd16k.bp"
# sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

# ARGS="4 $CYC $DIS $BLK SWAP QPD 8 io/isc26/rcs8-qpd16k.bp io/isc26/rcs12-qpd16k.bp"
# sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

# ARGS="4 $CYC $DIS $BLK SWAP QPD 12 io/isc26/rcs12-qpd16k.bp io/isc26/rcs16-qpd16k.bp"
# sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

# ARGS="4 $CYC $DIS $BLK SWAP QPD 16 io/isc26/rcs16-qpd16k.bp io/isc26/rcs20-qpd16k.bp"
# sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm
