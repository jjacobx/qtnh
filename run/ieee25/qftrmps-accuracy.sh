#!/bin/bash

WDIR=$QTNH_DIR/run
ARGS="-D $WDIR -o out/ieee25/qftrmps/%j.out"
SCRIPT=$WDIR/ieee25/run-qftzmps.slurm
EXPORTS="MPI_IMPL=UCX,EXP=ieee25-qftrmps-accuracy"

VAR="NSITES=40,CHIDIS=16,CHILOC_UP=16,CHILOC_DN=1,BDIM=16"
sbatch $ARGS -N 2 -n 256  --export=$VAR,$EXPORTS $SCRIPT

VAR="NSITES=40,CHIDIS=32,CHILOC_UP=16,CHILOC_DN=1,BDIM=32"
sbatch $ARGS -N 8 -n 1024 --export=$VAR,$EXPORTS $SCRIPT

