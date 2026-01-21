#!/bin/bash
set -e
pip install --break-system-packages tensorpool
tp cluster attach $(hostname) s-x9o8dimbyx --no-input
./setup.sh
mkdir -p /data/overhead_matching/{datasets,training_outputs}/
# copy datasets (took ~15 minutes)
rclone copy /mnt/flex-s-x9o8dimbyx/datasets/ /data/overhead_matching/datasets/ --transfers 8 --checkers 8 --ignore-checksum --progress

# extract any tar.gz files (parallel)
BASE_DIR="${1:-/data/overhead_matching/datasets}"

find "$BASE_DIR" -name '*.tar.gz' -print0 | xargs -0 -r -P 32 -I{} sh -c '
    echo "Extracting: $1"
    tar -xzf "$1" -C "$(dirname "$1")"
' _ {}



