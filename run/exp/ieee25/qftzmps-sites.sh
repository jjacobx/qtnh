#!/bin/bash

WDIR=$QTNH_DIR/run
SCRIPT=$WDIR/ieee25/run-qftzmps.slurm
EXPORTS="CHIDIS=1,CHILOC_UP=2,CHILOC_DN=2,BDIM=0,MPI_IMPL=UCX,EXP=ieee25-qftzmps-sites"

STEPS="0 200 400 600 800"

for s in ${STEPS}; do
  wait_running_jobs 40
  sbatch -D $WDIR -N 1 -n 1 --export=NSITES=$(($s+20)),$EXPORTS  $SCRIPT
  sbatch -D $WDIR -N 1 -n 1 --export=NSITES=$(($s+40)),$EXPORTS  $SCRIPT
  sbatch -D $WDIR -N 1 -n 1 --export=NSITES=$(($s+60)),$EXPORTS  $SCRIPT
  sbatch -D $WDIR -N 1 -n 1 --export=NSITES=$(($s+80)),$EXPORTS  $SCRIPT
  sbatch -D $WDIR -N 1 -n 1 --export=NSITES=$(($s+100)),$EXPORTS $SCRIPT
  sbatch -D $WDIR -N 1 -n 1 --export=NSITES=$(($s+120)),$EXPORTS $SCRIPT
  sbatch -D $WDIR -N 1 -n 1 --export=NSITES=$(($s+140)),$EXPORTS $SCRIPT
  sbatch -D $WDIR -N 1 -n 1 --export=NSITES=$(($s+160)),$EXPORTS $SCRIPT
  sbatch -D $WDIR -N 1 -n 1 --export=NSITES=$(($s+180)),$EXPORTS $SCRIPT
  sbatch -D $WDIR -N 1 -n 1 --export=NSITES=$(($s+200)),$EXPORTS $SCRIPT
done