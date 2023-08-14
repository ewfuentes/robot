#!/bin/env sh

set -x # enable echo

# Install clang14 and other required system packages
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install clang-15 clang-format-15 libxcursor-dev \
     libxrandr-dev libxinerama-dev libxi-dev freeglut3-dev libstdc++-12-dev gcc-11 libtbb-dev \
     libfmt-dev libspdlog-dev libvtk9-dev coinor-libipopt-dev coinor-libclp-dev
