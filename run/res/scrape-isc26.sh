#!/bin/bash

module load cray-R

EXPERIMENTS="rcs-setup rcs-block rcs-weak riqft-setup riqft-block riqft-strong riqft-fidelity"
ARGS="^[^T|] ^Targets ^$"

for exp in $EXPERIMENTS; do
  Rscript scrape.R isc26/$exp $ARGS
done
