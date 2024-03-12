#!/bin/bash
set -euox pipefail
echo $PWD
external/aarch64-none-linux-gnu/bin/aarch64-linux-gcc "$@" 
