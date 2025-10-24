#!/bin/bash

WDIR=$QTNH_DIR/run
SCRIPT=$WDIR/ieee25/run-qftsv.slurm
EXPORTS="MPI_IMPL=UCX,EXP=ieee25-qftsv-scaling"

LQ=16
wait_running_jobs 40
sbatch -D $WDIR -N 1  -n 16   --export=DQ=4,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 1  -n 32   --export=DQ=5,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 1  -n 64   --export=DQ=6,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 1  -n 128  --export=DQ=7,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 2  -n 256  --export=DQ=8,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 4  -n 512  --export=DQ=9,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 8  -n 1024 --export=DQ=10,LQ=${LQ},$EXPORTS $SCRIPT
sbatch -D $WDIR -N 16 -n 2048 --export=DQ=11,LQ=${LQ},$EXPORTS $SCRIPT

LQ=17
wait_running_jobs 40
sbatch -D $WDIR -N 1  -n 16   --export=DQ=4,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 1  -n 32   --export=DQ=5,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 1  -n 64   --export=DQ=6,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 1  -n 128  --export=DQ=7,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 2  -n 256  --export=DQ=8,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 4  -n 512  --export=DQ=9,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 8  -n 1024 --export=DQ=10,LQ=${LQ},$EXPORTS $SCRIPT
sbatch -D $WDIR -N 16 -n 2048 --export=DQ=11,LQ=${LQ},$EXPORTS $SCRIPT

LQ=18
wait_running_jobs 40
sbatch -D $WDIR -N 1  -n 16   --export=DQ=4,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 1  -n 32   --export=DQ=5,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 1  -n 64   --export=DQ=6,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 1  -n 128  --export=DQ=7,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 2  -n 256  --export=DQ=8,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 4  -n 512  --export=DQ=9,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 8  -n 1024 --export=DQ=10,LQ=${LQ},$EXPORTS $SCRIPT
sbatch -D $WDIR -N 16 -n 2048 --export=DQ=11,LQ=${LQ},$EXPORTS $SCRIPT

LQ=19
wait_running_jobs 40
sbatch -D $WDIR -N 1  -n 16   --export=DQ=4,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 1  -n 32   --export=DQ=5,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 1  -n 64   --export=DQ=6,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 1  -n 128  --export=DQ=7,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 2  -n 256  --export=DQ=8,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 4  -n 512  --export=DQ=9,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 8  -n 1024 --export=DQ=10,LQ=${LQ},$EXPORTS $SCRIPT
sbatch -D $WDIR -N 16 -n 2048 --export=DQ=11,LQ=${LQ},$EXPORTS $SCRIPT

LQ=20
wait_running_jobs 40
sbatch -D $WDIR -N 1  -n 16   --export=DQ=4,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 1  -n 32   --export=DQ=5,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 1  -n 64   --export=DQ=6,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 1  -n 128  --export=DQ=7,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 2  -n 256  --export=DQ=8,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 4  -n 512  --export=DQ=9,LQ=${LQ},$EXPORTS  $SCRIPT
sbatch -D $WDIR -N 8  -n 1024 --export=DQ=10,LQ=${LQ},$EXPORTS $SCRIPT
sbatch -D $WDIR -N 16 -n 2048 --export=DQ=11,LQ=${LQ},$EXPORTS $SCRIPT
