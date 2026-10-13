#!/usr/bin/env bash
# usage: run_carto_render.sh <region> <jobs.json> <out_dir> [nshards]
#   region:   docker volume suffix, e.g. massachusetts_260101 (see import_region.sh)
#   jobs.json: from make_carto_jobs.py
#   out_dir:  must live under VIGOR_ROOT (mounted rw at /vigor inside the container)
# Reuses the container ct2l-osm-run-<region> if it exists with the same VIGOR_ROOT mount,
# otherwise (re)creates it. Copies this dir's render_carto.py + the jobs file in, runs N
# shards, waits, prints wall time and tiles/s.
set -euo pipefail
REGION=$1; JOBS=$(realpath "$2"); OUT=$(realpath -m "$3"); N=${4:-16}
VIGOR_ROOT=${VIGOR_ROOT:-/data/overhead_matching/datasets/VIGOR}
IMAGE=overv/openstreetmap-tile-server:latest
NAME=ct2l-osm-run-$REGION
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case "$OUT" in "$VIGOR_ROOT"/*) ;; *) echo "out_dir $OUT is not under VIGOR_ROOT=$VIGOR_ROOT" >&2; exit 1;; esac
OUT_IN=/vigor${OUT#"$VIGOR_ROOT"}

docker volume inspect "ct2l-osm-db-$REGION" >/dev/null 2>&1 || { echo "no volume ct2l-osm-db-$REGION; run import_region.sh first" >&2; exit 1; }
if docker inspect "$NAME" >/dev/null 2>&1; then
  mounted=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/vigor"}}{{.Source}}:{{.RW}}{{end}}{{end}}' "$NAME")
  if [ "$mounted" != "$VIGOR_ROOT:true" ]; then echo "recreating $NAME (mount was '$mounted')"; docker rm -f "$NAME" >/dev/null; fi
fi
if ! docker inspect "$NAME" >/dev/null 2>&1; then
  docker run -d --name "$NAME" --shm-size=2g -e THREADS=8 \
    -v "ct2l-osm-db-$REGION:/data/database/" -v "$VIGOR_ROOT:/vigor" "$IMAGE" run >/dev/null
elif [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" != true ]; then
  docker start "$NAME" >/dev/null
fi
for _ in $(seq 1 120); do  # wait for postgres + mapnik.xml
  docker exec "$NAME" bash -c 'test -f /data/style/mapnik.xml && sudo -u renderer psql -d gis -c "select 1" >/dev/null 2>&1' && break
  sleep 2
done
docker exec "$NAME" mkdir -p /work
docker cp "$HERE/render_carto.py" "$NAME:/work/render_carto.py"
docker cp "$JOBS" "$NAME:/work/jobs.json"
mkdir -p "$OUT"

njobs=$(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$JOBS")
start=$(date +%s)
for i in $(seq 0 $((N-1))); do
  docker exec -u renderer "$NAME" python3 /work/render_carto.py /work/jobs.json --out-dir "$OUT_IN" --shard "$i/$N" &
done
wait
wall=$(( $(date +%s) - start ))
echo "$REGION: $njobs jobs, $N shards, ${wall}s wall, $(( njobs / (wall > 0 ? wall : 1) )) tiles/s -> $OUT"
