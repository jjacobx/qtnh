#!/bin/bash

WDIR=$QTNH_DIR/run
ARGS="-D $WDIR -o out/ieee25/qftrmps/%j.out"
SCRIPT=$WDIR/ieee25/run-qftzmps.slurm
EXPORTS="MPI_IMPL=UCX,EXP=ieee25-qftrmps-inout"

BDIMS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
VAR="NSITES=40,CHIDIS=16,CHILOC_UP=16,CHILOC_DN=16"
wait_running_jobs 35
for bd in ${BDIMS}; do
  sbatch $ARGS -N 2 -n 256 --export=BDIM=$bd,$VAR,$EXPORTS $SCRIPT
done

VAR="NSITES=80,CHIDIS=16,CHILOC_UP=16,CHILOC_DN=16"
wait_running_jobs 35
for bd in ${BDIMS}; do
  sbatch $ARGS -N 2 -n 256 --export=BDIM=$bd,$VAR,$EXPORTS $SCRIPT
done
