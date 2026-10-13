#!/usr/bin/env bash
# usage: import_region.sh <path/to/<region>-<yymmdd>.osm.pbf>
# Creates docker volume ct2l-osm-db-<region> (PBF stem, '-' -> '_') and runs the
# openstreetmap-tile-server osm2pgsql import into it. Idempotent: skips a volume that already
# holds a completed import. Volumes land in /var/lib/docker (~5 GB + 30x the PBF size).
set -euo pipefail
PBF=$(realpath "$1")
IMAGE=overv/openstreetmap-tile-server:latest
REGION=$(basename "$PBF" .osm.pbf | tr - _)
VOL=ct2l-osm-db-$REGION
if docker run --rm --entrypoint test -v "$VOL:/data/database/" "$IMAGE" -f /data/database/planet-import-complete 2>/dev/null; then
  echo "$VOL: import already complete"; exit 0
fi
need_gb=$(( 30 * $(stat -c %s "$PBF") / 1000000000 + 5 + 60 ))  # volume ~= 5 GB + 30x PBF
avail_gb=$(( $(df --output=avail -B1 / | tail -1) / 1000000000 ))  # docker root /var/lib/docker is on /
if [ "$avail_gb" -lt "$need_gb" ]; then echo "only ${avail_gb} GB free, need ~${need_gb} GB (5 GB + 30x PBF + 60 GB margin)" >&2; exit 1; fi
echo "importing $PBF -> $VOL (${avail_gb} GB free)"
start=$(date +%s)
docker volume create "$VOL" >/dev/null
docker run --rm --shm-size=1g -e THREADS=${THREADS:-8} -v "$PBF:/data/region.osm.pbf:ro" -v "$VOL:/data/database/" "$IMAGE" import
echo "$VOL: imported in $(( $(date +%s) - start ))s, $(docker system df -v | grep "$VOL" | awk '{print $NF}')"
