#!/bin/bash

testBin=$1
libPath=$2

. /opt/intel/oneapi/setvars.sh
export LD_LIBRARY_PATH=$libPath:$LD_LIBRARY_PATH

$testBin
