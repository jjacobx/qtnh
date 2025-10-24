#!/bin/bash

WDIR=$QTNH_DIR/run
SCRIPT=$WDIR/ieee25/run-qftzmps.slurm
EXPORTS="BDIM=0,MPI_IMPL=UCX,EXP=ieee25-qftzmps-chis"

N=40
CHILOCS="4 8 12 16 20 24 28 32"

for cl in ${CHILOCS}; do
  wait_running_jobs 40
  sbatch -D $WDIR -N 1 -n 16   --ntasks-per-node=16  --export=NSITES=$N,CHIDIS=4,CHILOC_UP=$cl,CHILOC_DN=$cl,$EXPORTS  $SCRIPT
  sbatch -D $WDIR -N 1 -n 64   --ntasks-per-node=64  --export=NSITES=$N,CHIDIS=8,CHILOC_UP=$cl,CHILOC_DN=$cl,$EXPORTS  $SCRIPT
  sbatch -D $WDIR -N 2 -n 144  --ntasks-per-node=72  --export=NSITES=$N,CHIDIS=12,CHILOC_UP=$cl,CHILOC_DN=$cl,$EXPORTS $SCRIPT
  sbatch -D $WDIR -N 2 -n 256  --ntasks-per-node=128 --export=NSITES=$N,CHIDIS=16,CHILOC_UP=$cl,CHILOC_DN=$cl,$EXPORTS $SCRIPT
  sbatch -D $WDIR -N 4 -n 400  --ntasks-per-node=100 --export=NSITES=$N,CHIDIS=20,CHILOC_UP=$cl,CHILOC_DN=$cl,$EXPORTS $SCRIPT
  sbatch -D $WDIR -N 6 -n 576  --ntasks-per-node=96  --export=NSITES=$N,CHIDIS=24,CHILOC_UP=$cl,CHILOC_DN=$cl,$EXPORTS $SCRIPT
  sbatch -D $WDIR -N 8 -n 784  --ntasks-per-node=98  --export=NSITES=$N,CHIDIS=28,CHILOC_UP=$cl,CHILOC_DN=$cl,$EXPORTS $SCRIPT
  sbatch -D $WDIR -N 8 -n 1024 --ntasks-per-node=128 --export=NSITES=$N,CHIDIS=32,CHILOC_UP=$cl,CHILOC_DN=$cl,$EXPORTS $SCRIPT
done
