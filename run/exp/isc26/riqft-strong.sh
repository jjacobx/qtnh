#!/bin/bash

BUILD_DIR=build-release
MPI_IMPL=UCX
TIME=2:00:0
EXP=riqft-strong
OUT=out/isc26/${EXP}/%j.out
PROG=examples/isc26/riqft

SITES=100
BLK=16
TOT=4096
SAT=0.1

MAX_LOC=512
MAX_DIS=32

pow2s() {
  local max=$1
  local i=1

  while :; do
    echo -n $i
    i=$((2 * $i))

    if [ $i -gt $max ]; then
      break
    fi

    echo -n " "
  done
}

DIS_VALS="$(pow2s $(($TOT / $BLK)))"
for dis in $DIS_VALS; do
  TASKS=$(($dis * $dis))
  TPN=128

  NODES=$(($TASKS / $TPN))
  if [ $NODES -eq 0 ]; then
    NODES=1
    TPN=$TASKS
  fi

  CYC=$(($TOT / ($dis * $BLK)))
  LOC=$(($CYC * $BLK))
  if [ $LOC -gt $MAX_LOC ]; then
    continue
  elif [ $dis -gt $MAX_DIS ]; then
    break
  fi

  ARGS="$SITES $CYC $dis $BLK $SAT QPD"
  sbatch -D $QTNH_DIR/run -N $NODES --ntasks-per-node=$TPN --time=$TIME -o $OUT \
    --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,EXP=$EXP,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
    $QTNH_DIR/run/run.slurm
done

# ARGS="$SITES 64 1 $BLK $SAT QPD"
# sbatch -D $QTNH_DIR/run -N 1 --ntasks-per-node=1 --time=$TIME -o $OUT \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,EXP=$EXP,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

# ARGS="$SITES 128 1 $BLK $SAT QPD"
# sbatch -D $QTNH_DIR/run -N 1 --ntasks-per-node=1 --time=$TIME -o $OUT \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,EXP=$EXP,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

# ARGS="$SITES 64 2 $BLK $SAT QPD"
# sbatch -D $QTNH_DIR/run -N 1 --ntasks-per-node=4 --time=$TIME -o $OUT \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,EXP=$EXP,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

# QOS=long
# TIME=96:00:0
# ARGS="$SITES 32 4 $BLK $SAT QPD"
# sbatch -D $QTNH_DIR/run -N 1 --ntasks-per-node=16 --time=$TIME -o $OUT --qos=$QOS \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,EXP=$EXP,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

# ARGS="$SITES 256 1 $BLK $SAT QPD"
# sbatch -D $QTNH_DIR/run -N 1 --ntasks-per-node=1 --time=$TIME -o $OUT \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,EXP=$EXP,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

# ARGS="$SITES 128 2 $BLK $SAT QPD"
# sbatch -D $QTNH_DIR/run -N 1 --ntasks-per-node=4 --time=$TIME -o $OUT \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,EXP=$EXP,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

# ARGS="$SITES 64 4 $BLK $SAT QPD"
# sbatch -D $QTNH_DIR/run -N 1 --ntasks-per-node=16 --time=$TIME -o $OUT \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,EXP=$EXP,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm

# QOS=long
# TIME=96:00:0
# ARGS="$SITES 32 8 $BLK $SAT QPD"
# sbatch -D $QTNH_DIR/run -N 1 --ntasks-per-node=64 --time=$TIME -o $OUT --qos=$QOS \
#   --export=BUILD_DIR=$BUILD_DIR,PROG=$PROG,EXP=$EXP,ARGS="$ARGS",MPI_IMPL=$MPI_IMPL \
#   $QTNH_DIR/run/run.slurm
