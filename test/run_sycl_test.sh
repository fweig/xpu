#!/bin/bash

testBin=$1
libPath=$2

. /opt/intel/oneapi/setvars.sh

$testBin
