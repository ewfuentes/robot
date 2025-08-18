#!/bin/env sh

set -x # enable echo

CODENAME=`lsb_release --codename --short`
if [ "${CODENAME}" = "noble" ]; then
    ln -s .bazelrc_noble .bazelrc_ubuntu
    PACKAGES="clang-18 clang-format-18 gcc-14 g++-14 gcc-11 g++-11"
elif [ "${CODENAME}" = "jammy" ]; then
    PACKAGES="clang-15 clang-format-15 gcc-12 g++-12 gcc-11 g++-11"
    ARCH=$(uname -m)
    if [ "${ARCH}" = "x86_64" ]; then
        ln -s .bazelrc_jammy .bazelrc_ubuntu
    elif [ "${ARCH}" = "aarch64" ]; then
        ln -s .bazelrc_jammy_arm64 .bazelrc_ubuntu
    else
        echo "Unsupported architecture: ${ARCH}"
        exit 1
    fi
fi


# Install clang14 and other required system packages
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install libxcursor-dev clang \
     libxrandr-dev libxinerama-dev libxi-dev freeglut3-dev libtbb-dev \
     libfmt-dev libspdlog-dev libvtk9-dev coinor-libipopt-dev coinor-libclp-dev \
     libgirepository1.0-dev libcairo2-dev libgtk2.0-dev libcanberra-gtk-module libsuitesparse-dev \
     python-is-python3 build-essential ${PACKAGES}

if [ "${CODENAME}" = "jammy" ]; then
    ARCH=$(uname -m)
    if [ "${ARCH}" = "aarch64" ]; then
        # We need to regenerate the python requirements. Use uv
        command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
        uv python install 3.12
        rm third_party/python/requirements_3_12.txt
        uv pip compile --python-version 3.12 -o third_party/python/requirements_3_12.txt \
            --format requirements.txt --index-strategy unsafe-best-match --generate-hashes \
            --emit-index-url third_party/python/requirements_3_12.in
    fi
fi

echo "Installed all packages. Ensure that cuda-toolkit is also installed!"
