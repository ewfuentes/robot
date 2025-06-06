#!/bin/env sh

set -x # enable echo

CODENAME=`lsb_release --codename --short`
if [ "${CODENAME}" = "noble" ]; then
    ln -s .bazelrc_noble .bazelrc_ubuntu
elif [ "${CODENAME}" = "jammy" ]; then
    ln -s .bazelrc_jammy .bazelrc_ubuntu
fi


# Install clang14 and other required system packages
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install clang clang-18 clang-format-18 libxcursor-dev \
     libxrandr-dev libxinerama-dev libxi-dev freeglut3-dev gcc-14 g++-14 gcc-11 g++-11 libtbb-dev \
     libfmt-dev libspdlog-dev libvtk9-dev coinor-libipopt-dev coinor-libclp-dev \
     libgirepository1.0-dev libcairo2-dev libgtk2.0-dev libcanberra-gtk-module libsuitesparse-dev \
     python-is-python3 build-essential

