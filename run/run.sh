#!/bin/bash

# Disable leak sanitier for MPI libs
export LSAN_OPTIONS=suppressions=lsan.supp,print_suppressions=0

PROG=qft
NQUBITS=20
DQUBITS=1
DIR=../build/examples
EXE="$DIR/$PROG $NQUBITS $DQUBITS"

NPROC=8
MPI_ARGS="-n $NPROC --oversubscribe"

mpirun $MPI_ARGS $EXE
