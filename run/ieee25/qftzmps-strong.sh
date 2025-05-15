#!/bin/bash

WDIR=$QTNH_DIR/run
SCRIPT=$WDIR/ieee25/run-qftzmps.slurm
EXPORTS="BDIM=0,MPI_IMPL=UCX,EXP=ieee25-qftzmps-strong"

N=40
CHITOTS="64 128 192 256 320"

for ct in ${CHITOTS}; do
  wait_running_jobs 40
  sbatch -D $WDIR -N 1  -n 1    --export=NSITES=$N,CHIDIS=1,CHILOC_UP=$(($ct / 1)),CHILOC_DN=$(($ct / 1)),$EXPORTS    $SCRIPT
  sbatch -D $WDIR -N 1  -n 4    --export=NSITES=$N,CHIDIS=2,CHILOC_UP=$(($ct / 2)),CHILOC_DN=$(($ct / 2)),$EXPORTS    $SCRIPT
  sbatch -D $WDIR -N 1  -n 16   --export=NSITES=$N,CHIDIS=4,CHILOC_UP=$(($ct / 4)),CHILOC_DN=$(($ct / 4)),$EXPORTS    $SCRIPT
  sbatch -D $WDIR -N 1  -n 64   --export=NSITES=$N,CHIDIS=8,CHILOC_UP=$(($ct / 8)),CHILOC_DN=$(($ct / 8)),$EXPORTS    $SCRIPT
  sbatch -D $WDIR -N 2  -n 256  --export=NSITES=$N,CHIDIS=16,CHILOC_UP=$(($ct / 16)),CHILOC_DN=$(($ct / 16)),$EXPORTS $SCRIPT
  sbatch -D $WDIR -N 8  -n 1024 --export=NSITES=$N,CHIDIS=32,CHILOC_UP=$(($ct / 32)),CHILOC_DN=$(($ct / 32)),$EXPORTS $SCRIPT
  sbatch -D $WDIR -N 32 -n 4096 --export=NSITES=$N,CHIDIS=64,CHILOC_UP=$(($ct / 64)),CHILOC_DN=$(($ct / 64)),$EXPORTS $SCRIPT
done
