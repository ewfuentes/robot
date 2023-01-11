#!/usr/bin/env sh
# Get the bazel root
INITIAL_DIR=`pwd`
BAZEL_ROOT=`bazel info | sed -nr "s/workspace: (.*)/\1/p"`

cd ${BAZEL_ROOT}
bazel build --noincompatible_use_python_toolchains --python_top experimental/pokerbots:system_runtime experimental/pokerbots:zipped_bot
cd ${INITIAL_DIR}
