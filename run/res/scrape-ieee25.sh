#!/bin/bash

module load cray-R

Rscript scrape.R ieee25/qftsv "^Contracted .* bonds$" "^Starting contraction" "^$"
Rscript scrape.R ieee25/qftn "^Contracted .* tensors$" "^$"
Rscript scrape.R ieee25/qftmps "^Iteration [0-9]+/[0-9]+$" "^$"
Rscript scrape.R ieee25/qftrmps "^Iteration [0-9]+/[0-9]+$" "^$"
Rscript scrape.R ieee25/rcsmps "^Iteration [0-9]+/[0-9]+$" "^(\\([0-9]+, [0-9]+\\)(, )?)+$" "^bonds = .*$" "^norm = .*$" "^$"
