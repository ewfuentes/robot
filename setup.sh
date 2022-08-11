#!/bin/env sh

set -x # enable echo

# Install clang14 and other required system packages
sudo apt install clang-14 libncurses-dev libxcursor-dev libxrandr-dev \
     libxinerama-dev libxi-dev freeglut3-dev
