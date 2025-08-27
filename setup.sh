#!/bin/env sh

set -x # enable echo

CODENAME=`lsb_release --codename --short`
ARCH=$(uname -m)
cat <<EOF > .bazelrc_ubuntu
build --config=clang
build:clang --platforms=//toolchain:${CODENAME}_clang_${ARCH}
build:gcc --platforms=//toolchain:${CODENAME}_gcc_${ARCH}
EOF

if [ "${CODENAME}" = "noble" ]; then
    PACKAGES="clang-18 clang-format-18 gcc-14 g++-14 gcc-11 g++-11"
elif [ "${CODENAME}" = "jammy" ]; then
    PACKAGES="clang-15 clang-format-15 gcc-12 g++-12 gcc-11 g++-11"
fi

# Install clang14 and other required system packages
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install libxcursor-dev clang \
     libxrandr-dev libxinerama-dev libxi-dev freeglut3-dev libtbb-dev \
     libfmt-dev libspdlog-dev libvtk9-dev coinor-libipopt-dev coinor-libclp-dev \
     libgirepository1.0-dev libcairo2-dev libgtk2.0-dev libcanberra-gtk-module libsuitesparse-dev \
     python-is-python3 build-essential ${PACKAGES}

if [ "${ARCH}" = "aarch64" ]; then
    # We need to regenerate the python requirements. Use uv
    command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv python install 3.12
    rm third_party/python/requirements_3_12.txt
    uv pip compile --python-version 3.12 -o third_party/python/requirements_3_12.txt \
        --format requirements.txt --index-strategy unsafe-best-match --generate-hashes \
        --emit-index-url third_party/python/requirements_3_12.in
fi

# Install bazelisk if not found
if command -v bazel > /dev/null 2>&1; then
    :
else
    if [ "${ARCH}" = "aarch64" ]; then
        BAZELISK_URL="https://github.com/bazelbuild/bazelisk/releases/download/v1.27.0/bazelisk-linux-arm64"
    elif [ "${ARCH}" = "x86_64" ]; then
        BAZELISK_URL="https://github.com/bazelbuild/bazelisk/releases/download/v1.27.0/bazelisk-linux-amd64"
    else
        echo "Unsupported architecture: ${ARCH}"
        exit 1
    fi
    mkdir -p ~/.local/bin
    curl -o ~/.local/bin/bazel -L "${BAZELISK_URL}"
    chmod +x ~/.local/bin/bazel
fi

echo "Installed all packages. Ensure that cuda-toolkit is also installed!"
