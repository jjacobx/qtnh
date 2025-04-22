#!/bin/bash

# QFTSV -- UCX is faster, maximum LQ=20
sbatch -D $QTNH_DIR/run -N 2 -n 256 --export=DQ=8,LQ=20,MPI_IMPL=UCX $QTNH_DIR/run/ieee25/run-qftsv.slurm

# QFTN -- UCX marginally faster
sbatch -D $QTNH_DIR/run -N 2 -n 256 --export=DQ=8,LQ=20,MULTI=3,MPI_IMPL=OFI $QTNH_DIR/run/ieee25/run-qftn.slurm