#!/bin/bash

BUILD_DIR=build
mkdir -p $BUILD_DIR
cd $BUILD_DIR

CMAKE_BUILD_TYPE=Release
CMAKE_PREFIX_PATH=/work/d35/d35/kubad35/local-libs
PROFILING=0
BUILD_TESTS=1

# cray/gnu/aocc
PRG_ENV=gnu
CXX=CC

module load PrgEnv-$PRG_ENV
[ $PROFILING == 1 ] && module load forge

cmake .. \
  -DCMAKE_CXX_COMPILER=${CXX} \
  -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
  -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH \
  -DMAP=$PROFILING \
  -DBUILD_TESTS=$BUILD_TESTS
