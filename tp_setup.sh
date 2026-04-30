#!/bin/bash
set -e
pip install tensorpool || pip install --break-system-packages tensorpool
VOLUME_ID="s-x9o8dimbyx"
if [ ! -d "/mnt/flex-${VOLUME_ID}" ]; then
    tp cluster attach $(hostname) "$VOLUME_ID" --wait
fi
./setup.sh
mkdir -p /data/overhead_matching/{datasets,training_outputs}/
# copy datasets (took ~15 minutes)
rclone copy /mnt/flex-s-x9o8dimbyx/torch ~/.cache/torch --transfers 8 --checkers 8 --ignore-checksum --progress
rclone copy /mnt/flex-s-x9o8dimbyx/datasets/VIGOR /data/overhead_matching/datasets/VIGOR --transfers 8 --checkers 8 --ignore-checksum --progress

# extract any tar.gz files that haven't been extracted yet (parallel)
BASE_DIR="${1:-/data/overhead_matching/datasets}"

find "$BASE_DIR" -name '*.tar.gz' -print0 | xargs -0 -r -P 32 -I{} sh -c '
    if [ ! -f "$1.extracted" ]; then
        echo "Extracting: $1"
        tar -xzf "$1" -C "$(dirname "$1")" && touch "$1.extracted"
    else
        echo "Already extracted: $1"
    fi
' _ {}



