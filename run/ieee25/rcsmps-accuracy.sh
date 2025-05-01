#!/bin/bash

WDIR=$QTNH_DIR/run
ARGS="-D $WDIR -o out/ieee25/rcsmps/%j.out"
SCRIPT=$WDIR/ieee25/run-rcsmps.slurm
EXPORTS="MPI_IMPL=UCX,EXP=ieee25-rcsmps-overlap"

# FVARS="NROW=4,NCOL=4,DEPTH=8,DCHI=8,LCHI_UP=64,LCHI_DN=1"
# sbatch $ARGS -N 1 -n 64 --export=$FVARS,$EXPORTS $SCRIPT

getn() { echo $((($1-1)/128+1)); }
nargs() { 
  PROCS=$(($1 * $1))
  NODES=$(getn $PROCS)
  echo "-N $NODES -n $PROCS"; 
}

# sbatch $ARGS $(nargs 16) --export=DCHI=16,LCHI_UP=8,$FVARS,$EXPORTS $SCRIPT

EXPORTS="MPI_IMPL=UCX,EXP=ieee25-rcsmps-accscaling"
LCHIS="32 16 8 4 2 1"

FVARS="NROW=4,NCOL=4,DEPTH=8,DCHI=32"
for x in ${LCHIS}; do
  sbatch $ARGS -N 8 -n 1024 --export=LCHI_UP=$x,$FVARS,$EXPORTS $SCRIPT
done

FVARS="NROW=5,NCOL=5,DEPTH=8,DCHI=32"
for x in ${LCHIS}; do
  sbatch $ARGS -N 8 -n 1024 --export=LCHI_UP=$x,$FVARS,$EXPORTS $SCRIPT
done

FVARS="NROW=6,NCOL=6,DEPTH=8,DCHI=32"
for x in ${LCHIS}; do
  sbatch $ARGS -N 8 -n 1024 --export=LCHI_UP=$x,$FVARS,$EXPORTS $SCRIPT
done

FVARS="NROW=7,NCOL=7,DEPTH=8,DCHI=32"
for x in ${LCHIS}; do
  sbatch $ARGS -N 8 -n 1024 --export=LCHI_UP=$x,$FVARS,$EXPORTS $SCRIPT
done

FVARS="NROW=7,NCOL=7,DEPTH=8,DCHI=48,LCHI_UP=48"
sbatch $ARGS -N 18 -n 2304 --export=$FVARS,$EXPORTS $SCRIPT
