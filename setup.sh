#!/bin/env sh

set -x # enable echo

# Install clang14 and other required system packages
sudo apt-get update
sudo apt-get install clang-14 libxcursor-dev libxrandr-dev \
     libxinerama-dev libxi-dev freeglut3-dev libstdc++-12-dev \
     gcc-11 libtbb-dev libfmt-dev libspdlog-dev libvtk9-dev \
