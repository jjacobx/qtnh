#!/bin/bash

WDIR=$QTNH_DIR/run
SCRIPT=$WDIR/ieee25/run-qftn.slurm
EXPORTS="MPI_IMPL=UCX,EXP=ieee25-qftn-groups"

DQ=6
wait_running_jobs 40

LQ=16
sbatch -D $WDIR -N 1 -n 64   --export=DQ=${DQ},LQ=${LQ},MULTI=1,$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 1 -n 64   --export=DQ=${DQ},LQ=${LQ},MULTI=2,$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 1 -n 64   --export=DQ=${DQ},LQ=${LQ},MULTI=3,$EXPORTS  $SCRIPT
LQ=18
sbatch -D $WDIR -N 1 -n 64   --export=DQ=${DQ},LQ=${LQ},MULTI=1,$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 1 -n 64   --export=DQ=${DQ},LQ=${LQ},MULTI=2,$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 1 -n 64   --export=DQ=${DQ},LQ=${LQ},MULTI=3,$EXPORTS  $SCRIPT
LQ=20
sbatch -D $WDIR -N 1 -n 64   --export=DQ=${DQ},LQ=${LQ},MULTI=1,$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 1 -n 64   --export=DQ=${DQ},LQ=${LQ},MULTI=2,$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 1 -n 64   --export=DQ=${DQ},LQ=${LQ},MULTI=3,$EXPORTS  $SCRIPT

DQ=8
wait_running_jobs 40

LQ=16
sbatch -D $WDIR -N 2 -n 256  --export=DQ=${DQ},LQ=${LQ},MULTI=1,$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 2 -n 256  --export=DQ=${DQ},LQ=${LQ},MULTI=2,$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 2 -n 256  --export=DQ=${DQ},LQ=${LQ},MULTI=3,$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 2 -n 256  --export=DQ=${DQ},LQ=${LQ},MULTI=4,$EXPORTS  $SCRIPT
LQ=18
sbatch -D $WDIR -N 2 -n 256  --export=DQ=${DQ},LQ=${LQ},MULTI=1,$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 2 -n 256  --export=DQ=${DQ},LQ=${LQ},MULTI=2,$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 2 -n 256  --export=DQ=${DQ},LQ=${LQ},MULTI=3,$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 2 -n 256  --export=DQ=${DQ},LQ=${LQ},MULTI=4,$EXPORTS  $SCRIPT
LQ=20
sbatch -D $WDIR -N 2 -n 256  --export=DQ=${DQ},LQ=${LQ},MULTI=1,$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 2 -n 256  --export=DQ=${DQ},LQ=${LQ},MULTI=2,$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 2 -n 256  --export=DQ=${DQ},LQ=${LQ},MULTI=3,$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 2 -n 256  --export=DQ=${DQ},LQ=${LQ},MULTI=4,$EXPORTS  $SCRIPT

DQ=10
wait_running_jobs 40

LQ=16
sbatch -D $WDIR -N 8 -n 1024 --export=DQ=${DQ},LQ=${LQ},MULTI=1,$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 8 -n 1024 --export=DQ=${DQ},LQ=${LQ},MULTI=2,$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 8 -n 1024 --export=DQ=${DQ},LQ=${LQ},MULTI=3,$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 8 -n 1024 --export=DQ=${DQ},LQ=${LQ},MULTI=4,$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 8 -n 1024 --export=DQ=${DQ},LQ=${LQ},MULTI=5,$EXPORTS  $SCRIPT
LQ=18
sbatch -D $WDIR -N 8 -n 1024 --export=DQ=${DQ},LQ=${LQ},MULTI=1,$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 8 -n 1024 --export=DQ=${DQ},LQ=${LQ},MULTI=2,$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 8 -n 1024 --export=DQ=${DQ},LQ=${LQ},MULTI=3,$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 8 -n 1024 --export=DQ=${DQ},LQ=${LQ},MULTI=4,$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 8 -n 1024 --export=DQ=${DQ},LQ=${LQ},MULTI=5,$EXPORTS  $SCRIPT
LQ=20
sbatch -D $WDIR -N 8 -n 1024 --export=DQ=${DQ},LQ=${LQ},MULTI=1,$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 8 -n 1024 --export=DQ=${DQ},LQ=${LQ},MULTI=2,$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 8 -n 1024 --export=DQ=${DQ},LQ=${LQ},MULTI=3,$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 8 -n 1024 --export=DQ=${DQ},LQ=${LQ},MULTI=4,$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 8 -n 1024 --export=DQ=${DQ},LQ=${LQ},MULTI=5,$EXPORTS  $SCRIPT
